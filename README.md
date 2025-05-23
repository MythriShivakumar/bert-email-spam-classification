# Email Spam Classification with BERT (Enron Dataset)

This project fine-tunes a BERT model on the Enron email dataset to perform binary classification of email content as spam or not spam. It uses the Hugging Face `transformers` and `datasets` libraries to streamline the process.

## 📬 Dataset

- **Source:** `SetFit/enron_spam` from Hugging Face
- Contains real-world emails labeled as **spam** or **ham (not spam)**

## 🧠 Model

- **Base Model:** `bert-base-uncased`
- **Tokenization:** Hugging Face `AutoTokenizer`
- Fine-tuned using PyTorch with a custom classification head

## 🧪 Metrics

- Accuracy
- Precision
- Recall
- F1-Score

## 🧰 Preprocessing

- Tokenized with padding and truncation (max_length = 512)
- Datasets split: 80% train, 10% val, 10% test
- Converted to PyTorch datasets with appropriate batching

## 🛠 Libraries Used

- `transformers`
- `datasets`
- `torch`
- `sklearn`

## 📈 Visualization & Output

- Metric scores on test set
- Optionally: Confusion matrix, sample predictions

## 🚀 How to Run

```bash
pip install transformers datasets
