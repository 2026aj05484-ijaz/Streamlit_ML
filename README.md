# Streamlit_ML
Adult Income Classification Dashboard.

# Adult Income Classification Dashboard Machine Learning Assignment 2

**Student:** IJAZ E  |  **ID:** 2026AJ05484
**Course:** M.Tech AIML – Machine Learning Assignment 2

## Problem statement
Build and evaluate five classification models that predict whether annual income is greater than $50K, then demonstrate the trained models through an interactive Streamlit application.

## Dataset description
The Adult Income dataset is from the UCI Machine Learning Repository. It contains demographic and employment attributes such as age, education, occupation, hours per week, capital gain/loss, and country. After removing missing records, it contains more than 30,000 instances and 14 input features, satisfying the assignment minimums. The target is binary: `0` for income <=50K and `1` for income >50K.

## Repository and application links
- GitHub Repository: **[https://github.com/2026aj05484-ijaz/Streamlit_ML]**
- Live Streamlit App: **[https://appml-o3ihgttsqizbbl4ghambuy.streamlit.app/]**

## Models used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Gaussian Naive Bayes
5. Support Vector Machine

The exact values are generated in `metrics.csv` after running `python train_models.py`.

## Project files
- `app.py` – Streamlit frontend
- `train_models.py` – download, preprocess, train, evaluate, and save models
- `test_data.csv` – held-out test data
- `metrics.csv` – metric comparison
- `model/` – saved pipelines and metadata
- `requirements.txt` – deployment dependencies

## How to run
```bash
python -m pip install -r requirements.txt
python train_models.py
streamlit run app.py
```
Upload `test_data.csv` in the application, select a model, and inspect the metrics, confusion matrix, and classification report. The application only loads saved models and does not retrain on uploaded data.

## Observations
Complete the model-specific observation table using `metrics.csv`. Select the overall winner using the highest F1/AUC/MCC while considering the balance between precision and recall.

## Academic integrity
This implementation is intended as a learning scaffold. Customize the UI, explanations, commit history, and observations, and cite the UCI dataset source before submission.
