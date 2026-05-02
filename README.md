# PDF AI Analyzer
The Smart Document Analyzer is a hybrid AI system that processes PDF documents and analyzes their content using machine learning.

It combines Python-based preprocessing with high-performance C++ inference using ONNX Runtime.

# Pipeline
PDF > Text Extraction (Python) > Chunking > Tokenization > tokens.json > C++ (ONNX Runtime) > Inference per chunk > Aggregated Document Result


# Features
* PDF text extraction
* Smart chunking for large documents
* AI-powered sentiment analysis
* Fast C++ inference engine
* Hybrid Python + C++ architecture
* Document-level aggregation
* (In progress) Section-level insights

# Tech Stack
* C++ (Core Engine)
* Python (Preprocessing)
* ONNX Runtime
* Hugging Face Transformers
* pdfplumber
* nlohmann/json

# Getting Started
1. Install Dependencies
   **Python**
   ```
   pip install torch transformers onnx onnxruntime pdfplumber onnxscript
   ```
**C++**
* Install Microsoft.ML.OnnxRuntime via NuGet
* Install nlohmann/json
