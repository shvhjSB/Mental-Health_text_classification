
> ⚠️ Trained model files and dataset are intentionally **not included** in the repository due to size and ethical considerations.

---

## 📊 Dataset

- Text-based dataset related to mental health expressions
- Used **only for training and experimentation**
- Dataset is **not redistributed** in this repository

**Reason:**
- Mental health data can be sensitive
- Best practice is to reference, not republish

---

## 🧹 Data Preprocessing

- Lowercasing text
- Removing punctuation and numbers
- Stopword removal using **NLTK**
- Handling missing values
- Class balancing using **RandomOverSampler**

---

## 🤖 Model Details

- **Base Model:** `bert-base-uncased`
- **Framework:** Hugging Face Transformers
- **Architecture:** `BertForSequenceClassification`
- **Loss:** Cross-Entropy Loss
- **Optimizer:** AdamW
- **Evaluation Metrics:**
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

---

## 🧪 Model Training

Training and evaluation were performed inside the Jupyter notebook using the Hugging Face `Trainer` API.

Key steps:
- Tokenization using BERT tokenizer
- Train-test split
- Model fine-tuning
- Performance evaluation

---

## 🖥️ Streamlit Application

The Streamlit app allows users to:
- Enter free-form text
- Perform real-time mental health classification
- View predicted mental health category

  
### Run locally:
```bash
pip install -r requirements.txt
streamlit run main.py
```


🛠️ Technologies Used

Python

Hugging Face Transformers

PyTorch

Scikit-learn

NLTK

Streamlit

Pandas, NumPy, Matplotlib, Seaborn

📌 Key Learnings

Practical fine-tuning of transformer models

Handling imbalanced text datasets

Deploying ML models using Streamlit

Managing large ML artifacts responsibly

Writing clean, reproducible ML projects


<img width="1905" height="952" alt="project_ss" src="https://github.com/user-attachments/assets/070ef344-1db5-46b5-942f-edc56ec5da82" />


