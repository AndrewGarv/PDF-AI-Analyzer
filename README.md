# PDF AI Analyzer
The Smart Document Analyzer is a hybrid AI system that processes PDF documents and analyzes their content using machine learning.

It combines Python-based preprocessing with high-performance C++ inference using ONNX Runtime.

# Pipeline
PDF
 ↓
Text Extraction (Python)
 ↓
Chunking
 ↓
Tokenization
 ↓
tokens.json
 ↓
C++ (ONNX Runtime)
 ↓
Inference per chunk
 ↓
Aggregated Document Result
