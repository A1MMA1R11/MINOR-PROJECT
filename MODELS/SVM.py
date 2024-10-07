import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("Breast_cancer_data.csv")
df
df.info()
df.describe()
df.isna().sum()
df.duplicated().sum()

num_cols = df.select_dtypes(include='number').columns
cat_cols = df.select_dtypes(include='object').columns

corr = df[num_cols].corr()
plt.figure(figsize=(30, 20))
sns.heatmap(corr, cmap='coolwarm', annot=True)

df[cat_cols].nunique()

df.hist(figsize=(20, 15))
df[cat_cols].nunique()

df['diagnosis'] = df.diagnosis.astype('category').cat.codes
df.diagnosis

#training and testing data
from sklearn.model_selection import train_test_split
X = df.drop("diagnosis", axis=1)
y = df.diagnosis

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=21)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

#model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

clf = make_pipeline(StandardScaler(), SVC())
clf.fit(X_train, y_train)
clf.score(X_train, y_train)
clf.score(X_test, y_test)

from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_estimator(
    clf, X_test, y_test, cmap='viridis')

#Predicting Results
y_pred = clf.predict(X_test)
#Calculating Resulta
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
print("Confusion Matrix: \n",confusion_matrix(y_test, y_pred))
print("accuracy score: {0}%".format(accuracy_score(y_test, y_pred) * 100))

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
rmacc = accuracy_score(y_test, y_pred)
print('Accuracy Score: ' + str(rmacc))
print('Precision Score: ' + str(precision_score(y_test, y_pred)))
print('Recall Score: ' + str(recall_score(y_test, y_pred)))
print('F1 Score: ' + str(f1_score(y_test, y_pred)))