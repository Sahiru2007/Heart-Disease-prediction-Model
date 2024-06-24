
---

# Heart Disease Prediction Model

This repository contains a Jupyter Notebook for building and evaluating a machine learning model to predict heart disease based on medical attributes. The model uses multiple classifiers and combines their predictions to provide a final diagnosis.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Data Preprocessing](#data-preprocessing)
- [Model Building](#model-building)
- [Model Evaluation](#model-evaluation)
- [Saving the Model](#saving-the-model)
- [Contributing](#contributing)
- [License](#license)

## Overview

The notebook builds and evaluates several machine learning models to predict heart disease. The steps include:

1. Loading and exploring the dataset.
2. Data preprocessing.
3. Splitting the data into training and testing sets.
4. Building multiple classifiers.
5. Evaluating the models using various metrics.
6. Saving the best model.

## Dataset

The dataset used in this notebook consists of medical attributes related to heart disease. Each row represents a patient with various attributes and the diagnosis result. The dataset can be found [here](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset).

### Dataset Description

The dataset contains the following columns:

- **age**: Age of the patient
- **sex**: Sex of the patient (1 = male, 0 = female)
- **cp**: Chest pain type (0-3)
- **trestbps**: Resting blood pressure (in mm Hg)
- **chol**: Serum cholesterol in mg/dl
- **fbs**: Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
- **restecg**: Resting electrocardiographic results (0-2)
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina (1 = yes, 0 = no)
- **oldpeak**: ST depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment (0-2)
- **ca**: Number of major vessels (0-3) colored by fluoroscopy
- **thal**: Thalassemia (1-3)
- **target**: Diagnosis of heart disease (1 = yes, 0 = no)

## Installation

To run this notebook, you need to have Python installed along with the following packages:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these packages using pip:

```sh
pip install pandas numpy matplotlib seaborn scikit-learn
```

Alternatively, you can use the provided `requirements.txt` file:

```sh
pip install -r requirements.txt
```

## Usage

1. Clone this repository:

```sh
git clone [<repository-url>](https://github.com/Sahiru2007/Heart-Disease-prediction-Model.git)
cd Heart-Disease-prediction-Model
```

2. Open the Jupyter Notebook:

```sh
jupyter notebook heart_disease_prediction.ipynb
```

3. Run all cells in the notebook to see the complete analysis and model evaluation.

## Data Preprocessing

### Handling Missing Values

Missing values in the dataset are handled by replacing them with the median of the respective columns.

### Feature Scaling

Standardization of features is done to bring all features to a similar scale:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)
```

### Splitting the Data

The dataset is split into training and testing sets using a 70-30 split:

```python
from sklearn.model_selection import train_test_split

X = heart_data.drop(columns='target', axis=1)
y = heart_data['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Model Building

### Models Used

The notebook evaluates several models:

- **Logistic Regression**
- **Random Forest**
- **Support Vector Machine (SVM)**

### Training the Models

Example: Training a Logistic Regression Classifier

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## Model Evaluation

### Metrics

The models are evaluated using the following metrics:

- **Accuracy**: The ratio of correctly predicted instances to the total instances.
- **Precision**: The ratio of true positive instances to the total predicted positives.
- **Recall**: The ratio of true positive instances to the actual positives.
- **F1 Score**: The harmonic mean of precision and recall.
- **Confusion Matrix**: A summary of prediction results on a classification problem.

### Example: Evaluating Logistic Regression Classifier

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'Confusion Matrix: \n{cm}')
```

### Evaluation Results

- **Logistic Regression**: Accuracy ~ 85%
- **Random Forest**: Accuracy ~ 88%
- **SVM**: Accuracy ~ 82%

## Unique Aspects of the Notebook

- **Correlation Matrix**: A heatmap to visualize the correlations between features.
- **Feature Importance Plot**: A plot to show the importance of each feature in the Random Forest model.
- **ROC Curve**: Receiver Operating Characteristic curve to evaluate the trade-off between sensitivity and specificity.

### Correlation Matrix

```python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = heart_data.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, fmt='.2f')
plt.title('Correlation Matrix')
plt.show()
```

### Feature Importance Plot

```python
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = heart_data.columns[:-1]

plt.figure(figsize=(12,6))
plt.title("Feature Importance")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.xlim([-1, X.shape[1]])
plt.show()
```

### ROC Curve

```python
from sklearn.metrics import roc_curve, auc

y_pred_prob = model.predict_proba(X_test)[:,1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
```

## Saving the Model

The best-performing model is saved using the `pickle` module for future use:

```python
import pickle

filename = 'heart_disease_model.pkl'
with open(filename, 'wb') as file:
    pickle.dump(model, file)

print(f"Model saved to {filename}")
```

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.

---


