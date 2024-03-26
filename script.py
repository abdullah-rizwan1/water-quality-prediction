import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
import torch.optim as optim


df = pd.read_csv('/Users/abdullahrizwan/Desktop/water-quality-prediction/water_potability.csv')

# dataframes ph values
print(df['ph'])

# Heatmap
sns.heatmap(df.isna(), yticklabels=False, cmap='crest')
plt.show()

indexes = df.index

# Histogram
solids = df['Solids']
plt.figure(figsize=(10,6))
plt.hist(indexes, bins=30, weights=solids, edgecolor='white')
plt.xlabel('Index')
plt.ylabel('Solids')
plt.title('Histogram of Solid values')
plt.show()

# Box Plot
hardness = df['Hardness']
plt.figure(figsize=(8,6))
plt.boxplot(hardness, vert=False)
plt.xlabel('Hardness Value')
plt.title('Box Plot of Hardness')
plt.show()

#Scatter Plot
chloramines = df['Chloramines']
plt.figure(figsize=(8,6))
plt.scatter(indexes, chloramines, alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Chloramines Value')
plt.title('Chloramines Values Scatter Plot')
plt.show()


# checking for null values
null_values = df.isnull().sum()
print("Null Values in Dataset")
print(null_values)


# Checking data types of values
df[df.columns].dtypes

# Replacing null values with the mean of each column
for column in df.columns:
    df[column] = df[column].fillna(df[column].mean())
    
df.isna().sum()