
# Comparative Study of Machine Learning Models for Breast Cancer Classification

## Project Overview

This project presents a comparative study of seven machine learning classification algorithms for breast cancer classification using the **Breast Cancer Wisconsin (Diagnostic) Dataset**.

The primary objective is to evaluate and compare different machine learning algorithms, understand their classification performance, and investigate whether hyperparameter tuning improves their predictive capabilities.

The study evaluates the models using multiple performance metrics, including Accuracy, Precision, Recall, F1-score, and ROC-AUC.

Both baseline models and hyperparameter-tuned models are compared to understand their strengths, limitations, and overall performance.

### Key Highlights

- Implemented and evaluated seven machine learning algorithms.
- Performed exploratory data analysis and data preprocessing.
- Applied feature scaling where required.
- Used GridSearchCV with 5-fold cross-validation for hyperparameter tuning.
- Compared baseline and tuned model performance.
- Visualized results using performance comparison charts.
- Analyzed model performance using multiple evaluation metrics.


## Problem Statement

Breast cancer classification is an important machine learning application in which diagnostic measurements are used to distinguish between malignant and benign tumours.

Different machine learning algorithms learn patterns in data differently. Therefore, comparing multiple algorithms helps us understand their strengths and limitations.

The objective of this project is to investigate how effectively different machine learning algorithms classify breast cancer cases and whether hyperparameter tuning can improve their performance.

## Project Objectives

1. Explore and understand the Breast Cancer Wisconsin (Diagnostic) dataset.
2. Perform exploratory data analysis to identify patterns and relationships among features.
3. Train seven different machine learning classification algorithms.
4. Evaluate baseline models using standard classification metrics.
5. Apply hyperparameter tuning using GridSearchCV and 5-fold cross-validation.
6. Compare baseline and tuned model performance.
7. Identify the best-performing models based on different evaluation metrics.
8. Analyze the impact of hyperparameter tuning on classification performance.

## Dataset Description

This project uses the **Breast Cancer Wisconsin (Diagnostic) Dataset**, available through the `scikit-learn` library.

The dataset contains diagnostic measurements computed from digitized images of fine needle aspirates (FNA) of breast masses.

The objective is to classify tumours into two categories:

- **Malignant (Class 0):** Cancerous tumour.
- **Benign (Class 1):** Non-cancerous tumour.

### Dataset Characteristics

| Characteristic | Description |
|---|---|
| Dataset Name | Breast Cancer Wisconsin (Diagnostic) |
| Source | Scikit-learn |
| Total Samples | 569 |
| Total Features | 30 |
| Target Classes | 2 |
| Missing Values | None |
| Feature Type | Numerical |
| Classification Type | Binary Classification |

### Feature Description

The dataset contains 30 numerical features describing characteristics of cell nuclei.

These features are derived from 10 basic measurements:

1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension

Each measurement is represented using three statistical values:

- **Mean:** Average measurement.
- **Standard Error:** Variation in the measurement estimate.
- **Worst:** Mean of the three largest measurements.

This results in a total of 30 numerical features.

### Target Distribution

The dataset contains:

- 212 malignant samples (Class 0).
- 357 benign samples (Class 1).

The dataset is moderately imbalanced, with more benign samples than malignant samples.

## Technologies and Libraries Used

### Programming Language
- Python

### Development Environment
- Visual Studio Code
- Jupyter Notebook

### Data Manipulation and Analysis
- **Pandas:** Data manipulation and tabular data analysis.
- **NumPy:** Numerical computations and array operations.

### Data Visualization
- **Matplotlib:** Creating charts and visualizing model performance.
- **Seaborn:** Exploratory data analysis and statistical visualizations.

### Machine Learning
- **Scikit-learn:** Data preprocessing, model implementation, evaluation metrics and hyperparameter tuning.
- **XGBoost:** Gradient boosting-based classification.

### Hyperparameter Optimization
- **GridSearchCV:** Systematic hyperparameter search using 5-fold cross-validation.


## Machine Learning Algorithms

We implemented and evaluated the following seven classification algorithms:

### 1. Logistic Regression
A supervised learning algorithm used for binary classification. It estimates the probability of a sample belonging to a particular class using the logistic function.

### 2. Decision Tree
A tree-based algorithm that makes predictions by splitting data based on feature values and learning decision rules.

### 3. Random Forest
An ensemble learning algorithm that combines multiple decision trees and aggregates their predictions to improve stability and generalization.

### 4. Support Vector Machine (SVM)
A supervised learning algorithm that identifies an optimal decision boundary, called a hyperplane, to separate different classes.

### 5. K-Nearest Neighbors (KNN)
A distance-based algorithm that classifies a sample based on the majority class among its nearest neighbors.

### 6. Gaussian Naive Bayes
A probabilistic classification algorithm based on Bayes' theorem. It assumes that features are conditionally independent given the class and follow Gaussian distributions.

### 7. XGBoost
An optimized gradient boosting algorithm that builds decision trees sequentially, with each new tree attempting to correct errors made by previous trees.



## Project Methodology

The project follows a systematic machine learning workflow consisting of the following stages:

### 1. Data Loading
- Loaded the Breast Cancer Wisconsin (Diagnostic) dataset using Scikit-learn.
- Separated the dataset into input features (X) and target labels (y).

### 2. Exploratory Data Analysis (EDA)
- Examined dataset dimensions, feature names and data types.
- Analyzed the distribution of malignant and benign samples.
- Investigated feature distributions and relationships.
- Visualized important patterns using statistical plots.

### 3. Data Preprocessing
- Checked for missing values and duplicate records.
- Separated features and target labels.
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used stratified sampling to preserve class proportions.
- Applied feature scaling to algorithms sensitive to feature magnitude.

### 4. Baseline Model Training
Implemented seven classification algorithms using their baseline configurations.

Each model was trained using the same training dataset and evaluated on the same unseen test dataset.

### 5. Model Evaluation
Evaluated baseline models using the following metrics:

- Accuracy
- Malignant Precision
- Malignant Recall
- F1-score
- ROC-AUC

Confusion matrices and performance visualizations were also generated to understand classification errors.

### 6. Hyperparameter Tuning
- Applied GridSearchCV to all seven classification algorithms.
- Used 5-fold cross-validation.
- Explored different hyperparameter combinations.
- Selected the configuration with the highest mean cross-validation accuracy for each model.

### 7. Tuned Model Evaluation
Evaluated the selected configurations on the same unseen test dataset using the same evaluation metrics.

### 8. Comparative Analysis
- Compared baseline and tuned model performance.
- Calculated metric-wise performance differences.
- Created visualizations to analyze improvements and declines.
- Ranked models based on their tuned F1-scores.
- Summarized the experimental findings.


## Experimental Results

All seven machine learning models were evaluated using the same test dataset containing 114 samples.

The models were compared before and after hyperparameter tuning using Accuracy, Malignant Precision, Malignant Recall, F1-score and ROC-AUC.

### 1. Baseline Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 98.25% | 97.62% | 97.62% | 97.62% | 0.9954 |
| Decision Tree | 91.23% | 84.78% | 92.86% | 88.64% | 0.9157 |
| Random Forest | 95.61% | 95.12% | 92.86% | 93.98% | 0.9937 |
| SVM | 97.37% | 95.35% | 97.62% | 96.47% | 0.9964 |
| KNN | 95.61% | 95.12% | 92.86% | 93.98% | 0.9788 |
| Gaussian Naive Bayes | 93.86% | 92.68% | 90.48% | 91.57% | 0.9878 |
| XGBoost | 94.74% | 95.00% | 90.48% | 92.68% | 0.9934 |

### 2. Tuned Model Performance

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 97.37% | 97.56% | 95.24% | 96.39% | 0.9957 |
| Decision Tree | 92.11% | 86.67% | 92.86% | 89.66% | 0.9163 |
| Random Forest | 95.61% | 95.12% | 92.86% | 93.98% | 0.9931 |
| SVM | 98.25% | 97.62% | 97.62% | 97.62% | 0.9937 |
| KNN | 97.37% | 100.00% | 92.86% | 96.30% | 0.9884 |
| Gaussian Naive Bayes | 93.86% | 92.68% | 90.48% | 91.57% | 0.9878 |
| XGBoost | 94.74% | 95.00% | 90.48% | 92.68% | 0.9921 |

**Note:** Precision, recall and F1-score refer to the malignant class (0). ROC-AUC is calculated using the probability of the malignant class. Scores are from one held-out test split.


## Hyperparameter Tuning Results

Hyperparameter tuning was performed using GridSearchCV with 5-fold cross-validation and accuracy as the optimization metric.

The following configurations achieved the highest mean cross-validation accuracy within their respective search grids.

| Model | Best Hyperparameters | Best CV Accuracy |
|---|---|---:|
| Logistic Regression | `C = 0.1` | 98.02% |
| Decision Tree | `max_depth = 5`, `min_samples_leaf = 1`, `min_samples_split = 5` | 93.85% |
| Random Forest | `n_estimators = 200`, `max_depth = None`, `max_features = sqrt` | 96.04% |
| SVM | `C = 0.1`, `kernel = linear`, `gamma = scale` | 97.80% |
| KNN | `n_neighbors = 7`, `weights = uniform`, `p = 2` | 97.14% |
| Gaussian Naive Bayes | `var_smoothing = 1e-09` | 93.63% |
| XGBoost | `n_estimators = 200`, `max_depth = 3`, `learning_rate = 0.1` | 97.58% |

### Cross-Validation Configuration

- Search strategy: Grid Search
- Cross-validation: 5 folds
- Optimization metric: Accuracy
- Model selection: Highest mean cross-validation accuracy

### Important Observation

The best hyperparameter configuration does not necessarily produce better performance on the held-out test dataset.

Hyperparameter tuning improved some models, while others showed unchanged or slightly lower test performance.

