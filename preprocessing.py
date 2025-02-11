import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv("Dataset/advertising.csv")

# Drop duplicates if any
df = df.drop_duplicates()

# Define features (X) and target variable (y)
X = df.drop(columns=['Sales'])  # Features
y = df['Sales']  # Target

# Normalize the feature variables
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert back to DataFrame with column names
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Combine processed features and target variable
processed_df = pd.concat([X_scaled_df, y], axis=1)

# Save to a new CSV file
processed_df.to_csv("Dataset/advertising_processed.csv", index=False)

print("Processed CSV file has been saved successfully as 'advertising_processed.csv'.")
