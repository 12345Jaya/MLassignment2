# Machine Learning Classification Model Comparison

## BITS Pilani WILP – Machine Learning Assignment 2

This project implements and compares five machine learning classification algorithms using the Dry Bean Dataset. A Streamlit web application is developed to evaluate trained models on an uploaded test dataset.

## Project Overview

The objective of this assignment is to build, evaluate, and compare multiple classification models using common performance metrics and provide an interactive web application for model evaluation.

### Dataset

- **Dataset:** Dry Bean Dataset
- **Source:** UCI Machine Learning Repository
- **Dataset link:** https://archive.ics.uci.edu/dataset/602/dry+bean+dataset
- **Number of instances:** 13,611
- **Number of features:** 16
- **Target variable:** Class
- **Number of classes:** 7

The dataset contains numerical features describing the shape and characteristics of dry beans.

## Machine Learning Algorithms

The following classification algorithms are implemented:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## Methodology

1. Load the Dry Bean Dataset.
2. Separate the input features and target variable.
3. Split the dataset into training and testing sets using an 80:20 ratio.
4. Apply feature scaling where required.
5. Train the five classification algorithms.
6. Evaluate the models using common classification metrics.
7. Save the trained models using Joblib.
8. Develop a Streamlit application to evaluate saved models on uploaded test data.

The train-test split uses `random_state=42` and stratification to maintain class distribution.

## Evaluation Metrics

The models are evaluated using the following metrics:

- **Accuracy:** Measures the proportion of correctly classified instances.
- **AUC:** Measures the ability of the model to distinguish between classes.
- **Precision:** Measures the correctness of positive predictions.
- **Recall:** Measures the ability to identify actual positive instances.
- **F1 Score:** Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC):** Measures the quality of classification predictions.

## Model Performance

The following results were obtained during model evaluation:

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9207 | 0.9934 | 0.9215 | 0.9207 | 0.9209 | 0.9042 |
| Decision Tree | 0.8917 | 0.9330 | 0.8912 | 0.8917 | 0.8913 | 0.8691 |
| KNN | 0.9166 | 0.9812 | 0.9174 | 0.9166 | 0.9168 | 0.8992 |
| Naive Bayes | 0.8979 | 0.9902 | 0.9007 | 0.8979 | 0.8981 | 0.8773 |
| SVM | 0.9221 | 0.9937 | 0.9224 | 0.9221 | 0.9222 | 0.9059 |

Based on the observed results, SVM achieved the highest accuracy and MCC among the five models.

## Streamlit Web Application

The application provides the following functionality:

- Selection of a trained machine learning model.
- Upload of a labelled test CSV file.
- Calculation of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Display of the confusion matrix.
- Display of the classification report.
- Download of predictions as a CSV file.

The application loads the saved model artifacts and evaluates the uploaded data without retraining the models.

## Project Structure

```text
MLassignment2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
├── .gitignore
│
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    └── metadata.json
```

## Installation and Execution

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/MLassignment2.git
cd MLassignment2
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit Application

```bash
streamlit run app.py
```

### 4. Access the Application

Open the local URL provided by Streamlit, usually:

```text
http://localhost:8501
```

### 5. Upload Test Data

Select a trained model and upload `test_data.csv` or another labelled CSV containing all required feature columns and the target column `Class`.

The application displays the evaluation metrics, confusion matrix, and classification report.

## Deployment

The Streamlit application can be deployed using Streamlit Community Cloud.

Deployment steps:

1. Connect the GitHub repository to Streamlit Community Cloud.
2. Select the repository and branch.
3. Specify `app.py` as the main application file.
4. Deploy the application.

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook
- Git and GitHub

## Conclusion

This project compares five machine learning classification algorithms on the Dry Bean Dataset. The performance of the models is analyzed using multiple evaluation metrics. An interactive Streamlit application is developed to make model evaluation accessible through a web interface.

The experimental results indicate that SVM performed best among the evaluated models based on accuracy and MCC.

## Author

BITS Pilani WILP – Machine Learning Assignment 2
