import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data_prep import load_and_clean

df = load_and_clean("./data/Telco-Customer-Churn.csv")

print(df.info())
print(df.describe())

print("Churn value counts:")
print(df['Churn'].value_counts())

plt.figure(figsize=(5,4))
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x='gender', hue='Churn', data=df)
plt.title("Churn by Gender")
plt.show()

plt.figure(figsize=(5,4))
sns.countplot(x='Contract', hue='Churn', data=df)
plt.title("Churn by Contract Type")
plt.show()

num_cols = df.select_dtypes(include=['float64','int64']).columns
plt.figure(figsize=(10,8))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
