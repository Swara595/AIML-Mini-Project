import pandas as pd

df = pd.read_csv("data/upi_transactions.csv")

print("Rows and columns:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nFraud vs normal:")
print(df["is_fraud"].value_counts())

print("\nFraud types:")
print(df["fraud_type"].value_counts())

print("\nAverage amount (normal vs fraud):")
print(df.groupby("is_fraud")["amount"].mean())
