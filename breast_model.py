import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score  # Add this import
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
import pickle
from datetime import datetime as dt
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load and explore dataset
df = pd.read_csv('Breast_cancer_data.csv')

# Splitting the dataset into features (X) and target (Y)
X = df.iloc[:, :3].values
Y = df.iloc[:, -1].values

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.175, random_state=0)

# Random Forest Classifier
st = dt.now()
randomforest = RandomForestClassifier(n_estimators=100, random_state=0)
randomforest.fit(X_train, Y_train)
print("Time taken to complete random search: ", dt.now() - st)

# Predict on the test set
random_pred = randomforest.predict(X_test)

# Calculate accuracy
rmacc = accuracy_score(Y_test, random_pred)  # Fix: accuracy_score was missing
print('Accuracy Score: ' + str(rmacc))

# Precision, recall, F1 score
print('Precision Score: ' + str(precision_score(Y_test, random_pred)))
print('Recall Score: ' + str(recall_score(Y_test, random_pred)))
print('F1 Score: ' + str(f1_score(Y_test, random_pred)))

# Save the trained model
filename = 'Breast_Cancer.sav'
pickle.dump(randomforest, open(filename, 'wb'))
