import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("Breast_cancer_data.csv")
df

df.info()

df.describe()

corr_mat = df.corr(numeric_only=True)
corr_mat = corr_mat.sort_values(by="diagnosis", ascending=False)
ordered_index = corr_mat.index
corr_mat = corr_mat.loc[:,ordered_index]
corr_mat

plt.figure(figsize=(20,18))
sns.heatmap(corr_mat, annot=True,fmt=".2f",)
plt.show()

sns.set_style("darkgrid")

features_to_graph = ordered_index[1:9]

num_features = len(features_to_graph)
num_rows = (num_features + 1) // 2

fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4.8 * num_rows))
# [6.4, 4.8]
for i, feature in enumerate(features_to_graph):
    ax_row = i // 2
    ax_col = i % 2
    sns.histplot(x=feature, data=df, hue="diagnosis", ax=axes[ax_row, ax_col])

plt.tight_layout()
plt.show()

features_to_graph = ordered_index[-4:]

num_features = len(features_to_graph)
num_rows = (num_features + 1) // 2
fig, axes = plt.subplots(num_rows, 2, figsize=(12, 4.8 * num_rows))
# [6.4, 4.8]

for i, feature in enumerate(features_to_graph):
    ax_row = i // 2
    ax_col = i % 2
    sns.histplot(x=feature, data=df, hue="diagnosis", ax=axes[ax_row, ax_col])

plt.tight_layout()
plt.show()