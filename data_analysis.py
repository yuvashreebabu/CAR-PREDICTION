import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("Dataset/advertising_processed.csv")

# Display basic information
print("Dataset Info:")
df.info()
print("\nSummary Statistics:")
print(df.describe())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Visualizing feature distributions
plt.figure(figsize=(10,5))
sns.histplot(df, kde=True, bins=30)
plt.title("Feature Distributions")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# Pairplot to visualize relationships
sns.pairplot(df)
plt.show()

# Boxplots to detect outliers
plt.figure(figsize=(10,5))
df.boxplot()
plt.title("Boxplot of Features")
plt.show()

print("Data Analysis Completed.")
