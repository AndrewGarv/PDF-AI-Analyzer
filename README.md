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


2. Generate Model (Python)
   ```
   python export_model.py
   ```
   Move files to
   ```
    /models/model.onnx
```
  

4. Process pdf
```
python process_pdf.py
```

4. Build and Run
Run Analyzer.exe

# Example Output
```
Model loaded successfully!
Processing chunk #0
Processing chunk #1
...

===== FINAL RESULT =====
Chunks analyzed: 11
Positive chunks: 8
Negative chunks: 3
Final Document Sentiment: POSITIVE
```


# Current Limitations
* Uses sentiment model (not semantic understanding)
* No cross-chunk context
* Requires Python preprocessing step
* Basic aggregation (count-based)

# Roadmap
* Highlight most important sections
* Add summarization model
* Confidence-weighted scoring
* Remove JSON (direct Python ↔ C++ bridge)
* Multi-document support
