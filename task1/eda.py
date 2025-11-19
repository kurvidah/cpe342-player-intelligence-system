# %% [markdown]
# # Task 1: Exploratory Data Analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('train_imputed.csv')

print(df.info())
print(df.head())

# %%
# Distribution plots for numerical features
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
for col in numerical_cols:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, hue='is_cheater', bins=30, kde=True, element="step", stat="density")
    plt.title(f"Distribution of '{col}'")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# %%
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
corr_mat = df[numerical_cols].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Matrix of Numerical Features")
plt.show()

# %%
# Top correlated features with 'is_cheater'
target_corr = corr_mat['is_cheater'].abs().sort_values(ascending=False)
print("Top correlated features with 'is_cheater':")
target_corr[1:16]

# %%
target_corr[1:16].index.tolist()


