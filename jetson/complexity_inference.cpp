// ─────────────────────────────────────────────────────────────────────────────
// icnet_inference.cpp
// ICNet complexity score inference using TensorRT
//
// Build:
//   g++ icnet_inference.cpp -o icnet_inference \
//       -I/usr/include/opencv4 \
//       -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
//       -lnvinfer -lcudart \
//       -std=c++17 -O2
//
// Usage:
//   ./icnet_inference <engine_file> <image_directory>
//   ./icnet_inference icnet_fp16.engine ./test_images/
// ─────────────────────────────────────────────────────────────────────────────

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <stdexcept>

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>

// ─────────────────────────────────────────────────────────────────────────────
// Preprocesssing constants — must match exactly what was used during training
// ─────────────────────────────────────────────────────────────────────────────
static const int   INPUT_H    = 512;
static const int   INPUT_W    = 512;
static const int   INPUT_C    = 3;

// ImageNet normalization (standard for models pretrained on ImageNet backbone)
// If your ICNet used different stats, change these values
static const float MEAN_R     = 0.485f;
static const float MEAN_G     = 0.456f;
static const float MEAN_B     = 0.406f;
static const float STD_R      = 0.229f;
static const float STD_G      = 0.224f;
static const float STD_B      = 0.225f;

// ─────────────────────────────────────────────────────────────────────────────
// CUDA error checking macro
// ─────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            throw std::runtime_error(                                           \
                std::string("CUDA error at line ")                              \
                + std::to_string(__LINE__)                                      \
                + ": "                                                          \
                + cudaGetErrorString(err));                                     \
        }                                                                       \
    } while (0)

// ─────────────────────────────────────────────────────────────────────────────
// TensorRT logger
// ─────────────────────────────────────────────────────────────────────────────
class Logger : public nvinfer1::ILogger
{
public:
    // Set to kWARNING to suppress info spam; kVERBOSE for debugging
    Severity reportableSeverity = Severity::kWARNING;

    void log(Severity severity, const char* msg) noexcept override
    {
        if (severity > reportableSeverity) return;

        switch (severity)
        {
            case Severity::kINTERNAL_ERROR: std::cerr << "[TRT INTERNAL_ERROR] "; break;
            case Severity::kERROR:          std::cerr << "[TRT ERROR] ";          break;
            case Severity::kWARNING:        std::cerr << "[TRT WARNING] ";        break;
            case Severity::kINFO:           std::cerr << "[TRT INFO] ";           break;
            case Severity::kVERBOSE:        std::cerr << "[TRT VERBOSE] ";        break;
            default:                        std::cerr << "[TRT UNKNOWN] ";        break;
        }
        std::cerr << msg << std::endl;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// RAII deleter for TensorRT objects
// TRT objects use destroy() instead of delete
// ─────────────────────────────────────────────────────────────────────────────
struct TRTDeleter
{
    template <typename T>
    void operator()(T* obj) const
    {
        if (obj) obj->destroy();
    }
};

template <typename T>
using TRTUniquePtr = std::unique_ptr<T, TRTDeleter>;

// ─────────────────────────────────────────────────────────────────────────────
// ICNet Inference Engine
// ─────────────────────────────────────────────────────────────────────────────
class ICNetInference
{
public:
    explicit ICNetInference(const std::string& enginePath)
    {
        loadEngine(enginePath);
        allocateBuffers();
        CUDA_CHECK(cudaStreamCreate(&stream_));

        std::cout << "─────────────────────────────────────────\n";
        std::cout << "ICNet TensorRT engine loaded\n";
        printBindingInfo();
        std::cout << "─────────────────────────────────────────\n";
    }

    ~ICNetInference()
    {
        // Free GPU buffers
        for (auto& buf : gpuBuffers_)
        {
            if (buf) cudaFree(buf);
        }
        if (stream_) cudaStreamDestroy(stream_);
    }

    // ── Run inference on a single image ──────────────────────────────
    // Returns complexity score in [0, 1]
    float infer(const cv::Mat& image)
    {
        // 1. Preprocess: resize → RGB → normalize → NCHW
        std::vector<float> inputData = preprocess(image);

        // 2. Copy input to GPU
        CUDA_CHECK(cudaMemcpyAsync(
            gpuBuffers_[inputBindingIdx_],
            inputData.data(),
            inputByteSize_,
            cudaMemcpyHostToDevice,
            stream_));

        // 3. Run inference
        bool ok = context_->enqueueV2(gpuBuffers_.data(), stream_, nullptr);
        if (!ok)
        {
            throw std::runtime_error("TensorRT enqueueV2 failed");
        }

        // 4. Copy output back to host
        std::vector<float> outputData(outputNumElements_);
        CUDA_CHECK(cudaMemcpyAsync(
            outputData.data(),
            gpuBuffers_[outputBindingIdx_],
            outputByteSize_,
            cudaMemcpyDeviceToHost,
            stream_));

        // 5. Sync before reading result
        CUDA_CHECK(cudaStreamSynchronize(stream_));

        // 6. ICNet outputs a single scalar — just return it
        // If your model outputs logits and needs sigmoid, apply it here:
        // return sigmoid(outputData[0]);
        return outputData[0];
    }

private:
    // ── TensorRT objects ──────────────────────────────────────────────
    Logger                            logger_;
    TRTUniquePtr<nvinfer1::ICudaEngine>       engine_;
    TRTUniquePtr<nvinfer1::IExecutionContext>  context_;
    cudaStream_t                      stream_ = nullptr;

    // ── Buffer management ─────────────────────────────────────────────
    std::vector<void*> gpuBuffers_;   // indexed by binding index
    int   inputBindingIdx_   = -1;
    int   outputBindingIdx_  = -1;
    size_t inputByteSize_    = 0;
    size_t outputByteSize_   = 0;
    size_t outputNumElements_= 0;

    nvinfer1::Dims inputDims_;
    nvinfer1::Dims outputDims_;

    // ── Load serialized engine from disk ─────────────────────────────
    void loadEngine(const std::string& path)
    {
        std::ifstream file(path, std::ios::binary);
        if (!file.good())
        {
            throw std::runtime_error("Cannot open engine file: " + path);
        }

        file.seekg(0, std::ios::end);
        const size_t fileSize = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> engineData(fileSize);
        file.read(engineData.data(), fileSize);
        if (!file)
        {
            throw std::runtime_error("Failed to read engine file: " + path);
        }

        TRTUniquePtr<nvinfer1::IRuntime> runtime(
            nvinfer1::createInferRuntime(logger_));
        if (!runtime)
        {
            throw std::runtime_error("Failed to create TensorRT runtime");
        }

        engine_.reset(runtime->deserializeCudaEngine(
            engineData.data(), fileSize));
        if (!engine_)
        {
            throw std::runtime_error("Failed to deserialize engine");
        }

        context_.reset(engine_->createExecutionContext());
        if (!context_)
        {
            throw std::runtime_error("Failed to create execution context");
        }
    }

    // ── Inspect bindings and allocate GPU memory ──────────────────────
    void allocateBuffers()
    {
        const int nbBindings = engine_->getNbBindings();
        gpuBuffers_.resize(nbBindings, nullptr);

        for (int i = 0; i < nbBindings; ++i)
        {
            const nvinfer1::Dims dims  = engine_->getBindingDimensions(i);
            const auto           dtype = engine_->getBindingDataType(i);

            // Only float32 output is handled here
            // (TRT fp16 engine still uses fp32 host buffers for I/O)
            size_t elemSize = 0;
            switch (dtype)
            {
                case nvinfer1::DataType::kFLOAT: elemSize = sizeof(float); break;
                case nvinfer1::DataType::kHALF:  elemSize = sizeof(float); break; // host side is fp32
                default:
                    throw std::runtime_error(
                        "Unsupported binding data type at index "
                        + std::to_string(i));
            }

            // Count total elements
            size_t numElems = 1;
            for (int d = 0; d < dims.nbDims; ++d)
            {
                numElems *= static_cast<size_t>(dims.d[d]);
            }
            const size_t byteSize = numElems * elemSize;

            CUDA_CHECK(cudaMalloc(&gpuBuffers_[i], byteSize));

            if (engine_->bindingIsInput(i))
            {
                inputBindingIdx_ = i;
                inputDims_       = dims;
                inputByteSize_   = byteSize;
            }
            else
            {
                outputBindingIdx_  = i;
                outputDims_        = dims;
                outputByteSize_    = byteSize;
                outputNumElements_ = numElems;
            }
        }

        if (inputBindingIdx_ == -1 || outputBindingIdx_ == -1)
        {
            throw std::runtime_error(
                "Engine must have exactly one input and one output binding");
        }
    }

    // ── Print binding shapes for debugging ────────────────────────────
    void printBindingInfo() const
    {
        auto printDims = [](const std::string& label,
                            const nvinfer1::Dims& dims)
        {
            std::cout << label << ": [";
            for (int i = 0; i < dims.nbDims; ++i)
            {
                std::cout << dims.d[i];
                if (i < dims.nbDims - 1) std::cout << ", ";
            }
            std::cout << "]\n";
        };

        printDims("Input  shape", inputDims_);
        printDims("Output shape", outputDims_);
        std::cout << "Input  buffer size: " << inputByteSize_  << " bytes\n";
        std::cout << "Output buffer size: " << outputByteSize_ << " bytes\n";
    }

    // ── Image preprocessing ───────────────────────────────────────────
    // Pipeline: BGR→RGB → resize to 512×512 → [0,1] → ImageNet normalize
    //           → NCHW float32 vector
    std::vector<float> preprocess(const cv::Mat& src) const
    {
        if (src.empty())
        {
            throw std::runtime_error("Empty image passed to preprocess()");
        }

        // Convert BGR (OpenCV default) to RGB
        cv::Mat rgb;
        if (src.channels() == 3)
        {
            cv::cvtColor(src, rgb, cv::COLOR_BGR2RGB);
        }
        else if (src.channels() == 1)
        {
            // Grayscale → RGB by repeating channel
            cv::cvtColor(src, rgb, cv::COLOR_GRAY2RGB);
        }
        else if (src.channels() == 4)
        {
            cv::cvtColor(src, rgb, cv::COLOR_BGRA2RGB);
        }
        else
        {
            throw std::runtime_error(
                "Unsupported number of channels: "
                + std::to_string(src.channels()));
        }

        // Resize to model input size
        cv::Mat resized;
        cv::resize(rgb, resized, cv::Size(INPUT_W, INPUT_H),
                   0, 0, cv::INTER_LINEAR);

        // Convert to float32 in [0, 1]
        cv::Mat floatImg;
        resized.convertTo(floatImg, CV_32FC3, 1.0 / 255.0);

        // Build NCHW tensor with ImageNet normalization
        // Layout: [1, C, H, W]
        const size_t planeSize = static_cast<size_t>(INPUT_H) * INPUT_W;
        std::vector<float> nchw(INPUT_C * planeSize);

        const float means[3] = { MEAN_R, MEAN_G, MEAN_B };
        const float stds[3]  = { STD_R,  STD_G,  STD_B  };

        for (int c = 0; c < INPUT_C; ++c)
        {
            float* dst = nchw.data() + c * planeSize;

            for (int h = 0; h < INPUT_H; ++h)
            {
                const cv::Vec3f* row = floatImg.ptr<cv::Vec3f>(h);
                for (int w = 0; w < INPUT_W; ++w)
                {
                    // floatImg channel order is RGB after cvtColor
                    dst[h * INPUT_W + w] =
                        (row[w][c] - means[c]) / stds[c];
                }
            }
        }

        return nchw;
    }

    // ── Optional sigmoid if model outputs raw logit ───────────────────
    static float sigmoid(float x)
    {
        return 1.0f / (1.0f + std::exp(-x));
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility: collect image paths from a directory
// ─────────────────────────────────────────────────────────────────────────────
std::vector<std::string> listImageFiles(const std::string& dirPath)
{
    static const std::vector<std::string> validExts =
        { ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".pgm" };

    std::vector<std::string> files;

    DIR* dir = opendir(dirPath.c_str());
    if (!dir)
    {
        std::cerr << "Failed to open directory: " << dirPath << std::endl;
        return files;
    }

    struct dirent* entry;
    while ((entry = readdir(dir)) != nullptr)
    {
        std::string name = entry->d_name;

        // Convert extension to lowercase for comparison
        std::string ext;
        auto dot = name.rfind('.');
        if (dot == std::string::npos) continue;
        ext = name.substr(dot);
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

        bool valid = std::any_of(
            validExts.begin(), validExts.end(),
            [&](const std::string& e){ return e == ext; });

        if (valid)
        {
            files.push_back(dirPath + "/" + name);
        }
    }
    closedir(dir);

    std::sort(files.begin(), files.end());
    return files;
}

// ─────────────────────────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────────────────────────
int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <engine_file> <image_directory>\n"
                  << "Example: " << argv[0]
                  << " icnet_fp16.engine ./test_images/\n";
        return EXIT_FAILURE;
    }

    const std::string engineFile = argv[1];
    const std::string imageDir   = argv[2];

    try
    {
        CUDA_CHECK(cudaSetDevice(0));

        ICNetInference icnet(engineFile);

        std::vector<std::string> images = listImageFiles(imageDir);
        if (images.empty())
        {
            std::cout << "No images found in: " << imageDir << "\n";
            return EXIT_SUCCESS;
        }

        std::cout << "Found " << images.size() << " image(s)\n";
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "\n";

        // Track stats
        float minScore  =  1e9f;
        float maxScore  = -1e9f;
        float sumScores =  0.0f;

        for (const auto& imgPath : images)
        {
            cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
            if (img.empty())
            {
                std::cerr << "[SKIP] Cannot read: " << imgPath << "\n";
                continue;
            }

            const float score = icnet.infer(img);

            // Clamp to [0,1] for display — model output should already be there
            const float clamped = std::max(0.0f, std::min(1.0f, score));

            // Simple bar visualisation
            const int   barLen  = 30;
            const int   filled  = static_cast<int>(clamped * barLen);
            std::string bar(filled, '#');
            bar += std::string(barLen - filled, '.');

            std::cout << "[" << bar << "] "
                      << std::setw(6) << score << "  "
                      << imgPath << "\n";

            minScore   = std::min(minScore,  score);
            maxScore   = std::max(maxScore,  score);
            sumScores += score;
        }

        // Summary
        std::cout << "\n─────────────────────────────────────────\n";
        std::cout << "Processed : " << images.size()  << " images\n";
        std::cout << "Min score : " << minScore        << "\n";
        std::cout << "Max score : " << maxScore        << "\n";
        std::cout << "Mean score: " << sumScores / images.size() << "\n";
        std::cout << "─────────────────────────────────────────\n";
    }
    catch (const std::exception& e)
    {
        std::cerr << "\nFATAL: " << e.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}