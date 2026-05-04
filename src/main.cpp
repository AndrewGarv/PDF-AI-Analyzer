#include <iostream>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <fstream>
#include <cmath>
#include <nlohmann/json.hpp>

struct ChunkResult {
    int index;
    std::string text;
    std::string label;
    float confidence;
};

using json = nlohmann::json;

int main() {
    // 1. Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);

    const wchar_t* model_path = L"models/model.onnx";
    Ort::Session session(env, model_path, session_options);

    std::cout << "Model loaded successfully!" << std::endl;

    // 2. Allocator + output name
    Ort::AllocatorWithDefaultOptions allocator;
    auto output_name_alloc = session.GetOutputNameAllocated(0, allocator);
    const char* output_name = output_name_alloc.get();

    std::cout << "Output name: " << output_name << std::endl;

    // 3. Load JSON 
    std::ifstream file("models/tokens.json");

    if (!file.is_open()) {
        std::cerr << "FAILED TO OPEN tokens.json" << std::endl;
        return 1;
    }

    json j;
    file >> j;

    std::cout << "Top-level JSON type: " << j.type_name() << std::endl;
    std::cout << "JSON size: " << j.size() << std::endl;

    if (!j.is_array()) {
        std::cerr << "ERROR: Expected JSON array!" << std::endl;
        return 1;
    }

    // 4. Memory info 
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);

    const char* input_names[] = { "input_ids", "attention_mask" };
    const char* output_names[] = { output_name };

    // 5. Aggregation counters
    int positive_count = 0;
    int negative_count = 0;

    int index = 0;

    std::cout << "Model inputs:" << std::endl;

    for (size_t i = 0; i < session.GetInputCount(); i++) {
        auto name_alloc = session.GetInputNameAllocated(i, allocator);
        auto type_info = session.GetInputTypeInfo(i);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();

        std::cout << "Input " << i << ": " << name_alloc.get() << " shape: ";

        for (auto dim : shape) {
            std::cout << dim << " ";
        }

        std::cout << std::endl;
    }

    // 6. Process each chunk
    for (const auto& chunk : j) {

        std::cout << "Processing chunk #" << index << std::endl;

        // Safety checks
        if (!chunk.is_object()) {
            std::cerr << "Skipping: chunk is not an object" << std::endl;
            index++;
            continue;
        }

        if (!chunk.contains("input_ids") || !chunk.contains("attention_mask")) {
            std::cerr << "Skipping: missing keys" << std::endl;
            index++;
            continue;
        }

        // This is for extracting the data
        std::vector<int64_t> input_ids;
        for (const auto& val : chunk["input_ids"]) {
            input_ids.push_back(static_cast<int64_t>(val));
        }

        std::vector<int64_t> attention_mask;
        for (const auto& val : chunk["attention_mask"]) {
            attention_mask.push_back(static_cast<int64_t>(val));
        }

        std::cout << "Chunk length: " << input_ids.size() << std::endl;

        // 7. Define shape per chunk
        std::vector<int64_t> input_shape = { 1, (int64_t)input_ids.size() };

        // 8. Create tensors
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            input_ids.data(),
            input_ids.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            attention_mask.data(),
            attention_mask.size(),
            input_shape.data(),
            input_shape.size()
        );

        Ort::Value input_tensors[] = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor)
        };

        // 9. Run inference
        std::vector<Ort::Value> output_tensors;

        try {
            output_tensors = session.Run(
                Ort::RunOptions{ nullptr },
                input_names,
                input_tensors,
                2,
                output_names,
                1
            );
        }
        catch (const Ort::Exception& e) {
            std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
            return 1;
        }

        float* output = output_tensors[0].GetTensorMutableData<float>();

        // 10. Softmax
        float exp0 = std::exp(output[0]);
        float exp1 = std::exp(output[1]);
        float sum = exp0 + exp1;

        float prob0 = exp0 / sum;
        float prob1 = exp1 / sum;

        // 11. Count results
        if (prob1 > prob0) {
            positive_count++;
        }
        else {
            negative_count++;
        }

        index++;
    }

    
    int total = positive_count + negative_count;

    std::cout << "\n===== FINAL RESULT =====" << std::endl;
    std::cout << "Chunks analyzed: " << total << std::endl;
    std::cout << "Positive chunks: " << positive_count << std::endl;
    std::cout << "Negative chunks: " << negative_count << std::endl;

    std::string final_label = (positive_count > negative_count) ? "POSITIVE" : "NEGATIVE";

    std::cout << "Final Document Sentiment: " << final_label << std::endl;

    return 0;
}
