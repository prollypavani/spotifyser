import pandas as pd
import numpy as np

df = pd.read_csv("dataset1.csv")

if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print("RAW DATA")
print(df.head())

print("SHAPE")
print(df.shape)

print("MISSING VALUES")
print(df.isnull().sum())

df = df.dropna()
df = df.drop_duplicates()

df.columns = df.columns.str.lower().str.replace(" ", "_")

print("CLEANED DATA")
print(df.head())

df.to_csv("dataset1_cleaned.csv", index=False)
