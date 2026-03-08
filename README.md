# Fake News Detection using Deep Learning (LSTM)

## Overview

This project builds a **Fake News Detection system** using Natural Language Processing (NLP) and Deep Learning.
The model classifies a news article as **Fake News** or **Real News**.

The system uses an **LSTM (Long Short-Term Memory) neural network** trained on a dataset of fake and real news articles.

---

## Project Workflow

1. Data Collection
2. Text Preprocessing
3. Tokenization and Sequence Padding
4. Deep Learning Model (LSTM)
5. Model Training and Evaluation
6. Prediction System
7. Interactive Interface for Testing

---

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Scikit-learn
* NLTK
* Gradio (for testing interface)

---

## Dataset

The dataset used in this project contains labeled **Fake News** and **Real News** articles.

Dataset source:

https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

Note:
The dataset is not included in this repository due to GitHub file size limitations.

---

## Model Architecture

Embedding Layer
↓
LSTM Layer
↓
Dense Output Layer (Sigmoid)

The sigmoid activation outputs a probability between **0 and 1**.

Prediction logic:

0 → Fake News
1 → Real News

---

## Model Performance

The trained model achieves approximately **98–99% accuracy** on the validation dataset.

---

## Project Structure

fake-news-detection/

notebook/
  fake_news_detection.ipynb

model/
  fake_news_model.h5
  tokenizer.pkl

app/
  gradio_app.py

requirements.txt
README.md

---

## How to Run the Project

### 1. Clone the repository

git clone https://github.com/your-username/fake-news-detection.git

### 2. Install dependencies

pip install -r requirements.txt

### 3. Run the application

python gradio_app.py

---

## Example Prediction

Input News Text:

The World Health Organization released a report highlighting the importance of vaccination programs worldwide.

Prediction:

Real News

---

## Screenshots

### Model Training

![Training Graph](training_graph.png)

### Prediction Interface

![Prediction App](images\Screenshot 2026-03-08 115158.png)

---

## Future Improvements

* Improve generalization on unseen news articles
* Implement **BERT-based fake news detection**
* Deploy as a web application
* Add explainable AI for prediction interpretation

---

## Author

**Y. Nanda Kishore Reddy**

Data Science Student | Machine Learning Enthusiast
