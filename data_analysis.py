import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("dataset/preprocessed_car_data.csv")

# Display basic statistics
print(df.describe())

# Check for correlations
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Feature Correlation Heatmap")
plt.show()

# Distribution of target variable
plt.figure(figsize=(8, 5))
sns.histplot(df['selling_price'], bins=50, kde=True)
plt.title("Distribution of Selling Price")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")
plt.show()

# Pairplot of selected features
selected_features = ['year', 'km_driven', 'mileage', 'engine', 'max_power', 'selling_price']
sns.pairplot(df[selected_features])
plt.show()

# Boxplot of selling price by fuel type
plt.figure(figsize=(8, 5))
sns.boxplot(x=df['fuel'], y=df['selling_price'])
plt.title("Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")
plt.show()

print("Data analysis completed!")
