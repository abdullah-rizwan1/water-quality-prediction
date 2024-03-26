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

# Removing Null Values
df.dropna(inplace=True)

indexes = df.index

# Histogram
ph_values = df['ph']
plt.figure(figsize=(10,6))
plt.hist(indexes, bins=30, weights=ph_values, edgecolor='white')
plt.xlabel('Index')
plt.ylabel('pH')
plt.title('Histogram of ph values')
plt.show()

# Box Plot
hardness = df['Hardness']
plt.figure(figsize=(8,6))
plt.boxplot(hardness, vert=False)
plt.xlabel('Hardness Value')
plt.title('Box Plot of Hardness')
plt.show()

#Scatter Plot
sulfate = df['Sulfate']
plt.figure(figsize=(8,6))
plt.scatter(indexes, sulfate, alpha=0.5)
plt.xlabel('Index')
plt.ylabel('Sulfate Value')
plt.title('Sulfate Values Scatter Plot')
plt.show()
