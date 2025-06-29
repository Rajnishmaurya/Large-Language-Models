# Large Language Models

A comprehensive collection of materials covering Large Language Models (LLMs), including theoretical lectures, hands-on practical exercises, and real-world applications.

## Overview

This repository contains a complete curriculum on Large Language Models, featuring:
- **Theoretical Foundations**: Detailed lecture materials covering LLM concepts, architectures, and applications
- **Hands-on Practice**: Interactive Jupyter notebooks with practical implementations
- **Real-world Applications**: Text classification, clustering, topic modeling, and more

## Repository Structure

```
Large-Language-Models/
├── Hands On Large Language Models/          # Practical Jupyter notebooks
│   ├── Hands_On_LLMS_Lec1_Introduction_to_Transformers.ipynb
│   ├── Hands_On_LLMS_Chapter2_Tokens_and_token_Embeddings.ipynb
│   ├── Hands_on_LLMS_Lec3_Looking_Inside_Transformers_LLMs.ipynb
│   ├── Hands_On_LLM_Lecture_4_Text_Classification.ipynb
│   └── Hands_On_LLMs_Lecture_5_Text_Clustering_and_Topic_Modelling.ipynb
├── lecture/                                 # Theoretical lecture materials
│   ├── lcs2/                               # Core LLM concepts (PDFs)
│   ├── week1/ - week12/                    # Weekly lecture materials
│   └── Week5/ - Week11/                    # Advanced topics
└── README.md
```

##  Learning Path

### 1. Foundations (Week 1-4)
- **Introduction to Transformers**: Understanding the transformer architecture
- **Tokens and Embeddings**: Deep dive into tokenization and embeddings
- **Core Concepts**: Basic LLM principles and terminology

### 2. Advanced Topics (Week 5-8)
- **Inside Transformers**: Detailed exploration of transformer internals
- **Text Classification**: Practical applications of LLMs for classification tasks
- **Model Architecture**: Understanding different LLM architectures

### 3. Applications (Week 9-12)
- **Text Clustering**: Unsupervised learning with LLMs
- **Topic Modeling**: Extracting themes and topics from text
- **Real-world Applications**: Industry use cases and implementations

##  Hands-on Exercises

### Prerequisites
```bash
pip install transformers>=4.40.1
pip install accelerate>=0.27.2
pip install sentence-transformers>=3.0.1
pip install gensim>=4.3.2
pip install scikit-learn>=1.5.0
```

### Key Notebooks

1. **Introduction to Transformers** (`Hands_On_LLMS_Lec1_Introduction_to_Transformers.ipynb`)
   - Loading and using pre-trained models (Microsoft Phi-3)
   - Text generation with transformers
   - Basic model interaction

2. **Tokens and Embeddings** (`Hands_On_LLMS_Chapter2_Tokens_and_token_Embeddings.ipynb`)
   - Understanding tokenization across different models
   - Visualizing token embeddings
   - Comparing BERT, GPT-2, and T5 tokenizers

3. **Text Classification** (`Hands_On_LLM_Lecture_4_Text_Classification.ipynb`)
   - Building text classification models
   - Fine-tuning pre-trained models
   - Evaluation and performance metrics

4. **Text Clustering and Topic Modeling** (`Hands_On_LLMs_Lecture_5_Text_Clustering_and_Topic_Modelling.ipynb`)
   - Unsupervised text analysis
   - Topic extraction techniques
   - Clustering algorithms with embeddings

## Lecture Materials

The `lecture/` directory contains comprehensive PDF materials organized by weeks:

- **lcs2/**: Core LLM concepts and fundamentals
- **week1/ - week12/**: Progressive learning from basics to advanced topics
- **Week5/ - Week11/**: Specialized topics and advanced applications

## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Large-Language-Models
   ```

2. **Set up your environment**:
   ```bash
   # Create a virtual environment (recommended)
   python -m venv llm_env
   source llm_env/bin/activate  # On Windows: llm_env\Scripts\activate
   ```

3. **Start with the basics**:
   - Read through the lecture materials in `lecture/week1/`
   - Run the first notebook: `Hands_On_LLMS_Lec1_Introduction_to_Transformers.ipynb`

4. **Follow the learning path**:
   - Complete notebooks in order
   - Read corresponding lecture materials
   - Experiment with different models and parameters

##  Technical Requirements

- **Python**: 3.8 or higher
- **GPU**: Recommended for faster model training (CUDA-compatible)
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: 5GB+ free space for models and datasets

##  Models Used

- **Microsoft Phi-3-mini-4k-instruct**: Primary model for demonstrations
- **BERT variants**: For tokenization examples
- **GPT-2**: For text generation examples
- **T5/Flan-T5**: For sequence-to-sequence tasks

---
