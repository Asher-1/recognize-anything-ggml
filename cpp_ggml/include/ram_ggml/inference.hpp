#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ram_ggml {

enum class Backend { Cpu, Cuda, Vulkan };
enum class Task { Tags, Caption };
enum class CudaCompute { Auto, F32, F16 };

// All process configuration belongs here. The CLI is only an adapter that
// fills this object, so adding a runtime option does not grow a call signature.
class InferenceOptions {
public:
    std::string model_path = "cpp_ggml/models/gguf/ram_swin_large_14m-f16.gguf";
    std::string image_path = "images/demo/demo1.jpg";
    std::string input_bin;
    std::string vocab_path;
    std::string specified_tags;
    // Optional label-space adapter. The embedding file is raw float32 with
    // 512 columns; RAM++ stores one or more descriptions per class.
    std::string tag_list_path;
    std::string threshold_path;
    std::string label_embedding_path;
    float threshold_override = -1.0f;
    // Compensates for backend/quantization ulp drift at Tag2Text's published
    // class threshold without changing RAM/RAM++ label selection.
    float threshold_epsilon = 0.0f;
    std::string dump_dir;
    std::string logits_out;
    // When set, load the model once and process one image path per line.
    std::string manifest_path;
    std::string jsonl_path;
    std::string logits_matrix_path;
    int max_images = 0; // zero means all manifest rows
    Backend backend = Backend::Cpu;
    Task task = Task::Tags;
    CudaCompute cuda_compute = CudaCompute::Auto;
    int threads = 8;
    int warmup = 3;
    int repeat = 3;
    int max_length = 30;
    int beam_batch = 0; // zero selects 1 on CPU and 3 on accelerators
    bool pipeline = false;
    bool profile_backends = false;
};

struct InferenceResult {
    std::vector<float> logits;
    std::string caption;
    std::string effective_cuda_compute = "n/a";
    double latency_ms = 0.0;
};

class InferenceRunner {
public:
    static bool run(const InferenceOptions & options, InferenceResult * result = nullptr,
                    std::string * error = nullptr);
};

} // namespace ram_ggml
