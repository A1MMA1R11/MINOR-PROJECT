import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

import plotly.graph_objects as go

df = pd.read_csv("Breast_cancer_data.csv")

#training and testing data
from sklearn.model_selection import train_test_split
X = df.drop("diagnosis", axis=1)
y = df.diagnosis
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, shuffle=True, random_state=21)
X_train.shape, y_train.shape, X_test.shape, y_test.shape

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#LR model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

y_proba_1 = classifier.predict_proba(X_test)
fpr_1, tpr_1, thresholds = roc_curve(y_test, y_proba_1[:, 1])

thresholds = thresholds[thresholds < 1]

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
randomforest = RandomForestClassifier(n_estimators = 100,random_state = 0)
randomforest.fit(X_train, y_train)
random_pred = randomforest.predict(X_test)

y_proba_2 = randomforest.predict_proba(X_test)
fpr_2, tpr_2, thresholds = roc_curve(y_test, y_proba_2[:, 1])

#DT Model
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(max_depth=5)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

y_proba_3 = dt.predict_proba(X_test)
fpr_3, tpr_3, thresholds = roc_curve(y_test, y_proba_3[:, 1])

#Naive Bayes model
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred= clf.predict(X_test)

y_proba_4 = clf.predict_proba(X_test)
fpr_4, tpr_4, thresholds = roc_curve(y_test, y_proba_4[:, 1])

#KNN classifier model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
clf = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=3))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_proba_5 = clf.predict_proba(X_test)
fpr_5, tpr_5, thresholds = roc_curve(y_test, y_proba_5[:, 1])

#SVM model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
#clf = SVC(kernel='rbf', C=1e9, gamma=1e-07, probability=True).fit(xtrain,ytrain)
clf = make_pipeline(StandardScaler(), SVC(probability= True))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

y_proba_6 = clf.predict_proba(X_test)
fpr_6, tpr_6, thresholds = roc_curve(y_test, y_proba_6[:, 1])


fig = go.Figure()
fig.add_trace(go.Scatter(x=fpr_1, y=tpr_1,
                    mode='lines+markers',
                    name='ROC Curve for the LR model'))
fig.add_trace(go.Scatter(x=fpr_2, y=tpr_2,
                    mode='lines+markers',
                    name='ROC Curve for the RandomForest model'))
fig.add_trace(go.Scatter(x=fpr_3, y=tpr_3,
                    mode='lines+markers',
                    name='ROC Curve for the DT model'))
fig.add_trace(go.Scatter(x=fpr_4, y=tpr_4,
                    mode='lines+markers',
                    name='ROC Curve for the Naive Bayes model'))
fig.add_trace(go.Scatter(x=fpr_5, y=tpr_5,
                    mode='lines+markers',
                    name='ROC Curve for the KNN model'))
fig.add_trace(go.Scatter(x=fpr_6, y=tpr_6,
                    mode='lines+markers',
                    name='ROC Curve for the SVM model'))
fig.update_layout(
    title="ROC Curve",
    xaxis_title="FPR",
    yaxis_title="FPR",
)
fig.show()