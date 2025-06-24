# IS-584 Term Project: Review-Based Paper Acceptance Classification

This project implements a parameter-efficient classification model for predicting paper acceptance decisions based on metareviews produced using peer review texts using the ASAP-Review dataset. The final architecture uses a quantized TinyLLaMA-1.1B-Chat model fine-tuned with LoRA adapters, followed by interpretability analysis using SHAP. TermProjectFineTune.ipynb is trained using Google Collab with 

## 🔍 Weights & Biases (W&B) Tracking

All training and evaluation experiments were tracked using [Weights & Biases](https://wandb.ai).

> **Note:** Due to organization-level restrictions, I am unable to publicly share the online W&B project dashboards.

However, all run logs and artifacts are available locally under the `wandb/` directory included in this repository. You can inspect training metrics, configuration, and model performance by navigating through these logs. The report can be seen from the [link](https://wandb.ai/e274028-metu-middle-east-technical-university/tinyllama-lora/reports/Final-Report--VmlldzoxMzMzNzY0NQ?accessToken=9p332uvtlizc244x4ri0kckew4qka46hmor1zi128m49jnw4cyptt4qru8jq9v8z)

---

## 📁 Project Structure

IS-584-TERM-PROJECT/
├── data/ # dataset.zip (not committed due to GitHub size limits)
├── notebooks/ # All Jupyter notebooks used for development
│ ├── Baselines.ipynb # TF-IDF, Logistic Regression, SBERT baselines
│ ├── few-shot-learning.ipynb # Prompt-based classification experiments
│ ├── PreliminaryResults.ipynb # Initial exploratory runs
│ └── TermProjectFineTune.ipynb # Final LoRA fine-tuning and SHAP analysis
├── wandb/ # Weights & Biases logs
├── requirements.txt # Python package requirements
├── .gitignore # Git ignore rules
└── README.md # You're here!


---

## 🧠 Model Overview

- **Backbone**: TinyLLaMA-1.1B-Chat-v1.0 (4-bit quantized)
- **PEFT**: LoRA adapters applied to layers 12–21
- **Classifier**: Linear head on pooled token embeddings
- **Training**: Hugging Face `Trainer` with `wandb` sweep support
- **Metrics**: Accuracy, Precision, Recall, F1, AUC

---

## 🧪 Interpretability

Interpretability was performed using the SHAP library, applied to the final classifier head after pooling the model's hidden states. Token-level contribution scores were visualized for selected review examples.

---

## 🗂️ Notebooks

- `Baselines.ipynb`: Implements majority class, TF-IDF + LR, SBERT + LR, and SBERT + MLP.
- `few-shot-learning.ipynb`: Explores prompt-based classification using in-context examples.
- `TermProjectFineTune.ipynb`: Core training script using LoRA + Trainer + W&B.
- `PreeliminaryResults.ipynb`: Initial experiments with baseline classifiers and early LoRA runs.

---

## 📦 Installation

pip install -r requirements.txt

--- 

### 📚 Dataset Information

**Dataset Name:** ASAP-Review  
**Source:** [Kaggle - ASAP-Review Dataset](https://www.kaggle.com/datasets/jonauskis/asap-review)  

**Description:**  
The ASAP-Review dataset consists of peer reviews and metadata for ICLR and NeurIPS submissions. It includes:
- Full review texts
- Review scores and confidence levels
- metareviews

This dataset supports tasks like:
- Paper acceptance prediction  
- Review quality modeling  
- Sentence-level aspect classification  

---

### 🧑‍💻 Project Author

**Project Title:** IS-584 Term Project - Paper Acceptance Classification from Peer Reviews  
**Author:** Çağrı Gülbeycan  
**Institution:** Middle East Technical University  
**Email:**  
- e274028@metu.edu.tr 
**Year:** 2025

