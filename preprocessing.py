import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("dataset/Car details v3.csv")

# Drop irrelevant columns
df.drop(columns=['name', 'torque'], inplace=True)

# Handling missing values
df['mileage'] = df['mileage'].str.extract(r'(\d+\.\d+)').astype(float)
df['engine'] = df['engine'].str.extract(r'(\d+)').astype(float)
df['max_power'] = df['max_power'].str.extract(r'(\d+\.\d+)').astype(float)

imputer = SimpleImputer(strategy='median')
df[['mileage', 'engine', 'max_power', 'seats']] = imputer.fit_transform(df[['mileage', 'engine', 'max_power', 'seats']])

# Encoding categorical features
label_enc = LabelEncoder()
df['fuel'] = label_enc.fit_transform(df['fuel'])
df['seller_type'] = label_enc.fit_transform(df['seller_type'])
df['transmission'] = label_enc.fit_transform(df['transmission'])
df['owner'] = label_enc.fit_transform(df['owner'])

# Feature Scaling
scaler = StandardScaler()
df[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']] = scaler.fit_transform(df[['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats']])

# Save preprocessed dataset
df.to_csv("dataset/preprocessed_car_data.csv", index=False)

print("Data preprocessing completed. Saved as 'preprocessed_car_data.csv'")