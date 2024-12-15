# /// script
# requires-python = ">=3.9"
# dependencies = [
#   "pandas",
#   "seaborn",
#   "matplotlib",
#   "numpy",
#   "scipy",
#   "openai",
#   "scikit-learn",
#   "requests",
#   "ipykernel",  # Added ipykernel
# ]
# ///

import os
import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import argparse
import requests
import json
import openai

def analyze_data(df):
    # 1. Basic Dataset Information
    print("📝 Basic Dataset Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.info())

    # 2. Summary Statistics
    print("\n📊 Summary Statistics:")
    print(df.describe())  

    # 3. Missing Values
    print("\n🔍 Missing Values per Column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 4. Correlation Matrix
    print("\n📈 Correlation Matrix (Numerical Columns):")
    correlation_matrix = df.select_dtypes(include=[np.number]).corr()
    print(correlation_matrix)

    # 5. Outlier Detection (IQR Method)
    print("\n🚨 Outlier Detection:")
    def detect_outliers(col):
        Q1 = col.quantile(0.25)
        Q3 = col.quantile(0.75)
        IQR = Q3 - Q1
        outliers = col[(col < Q1 - 1.5 * IQR) | (col > Q3 + 1.5 * IQR)]
        return len(outliers)

    numeric_cols = df.select_dtypes(include=[np.number])
    for col in numeric_cols:
        num_outliers = detect_outliers(numeric_cols[col])
        print(f"{col}: {num_outliers} outliers detected")

    # 6. Clustering Suggestion (for large datasets)
    print("\n🔗 Clustering Suggestions:")
    if len(df) > 1000:
        print("Dataset is large. Consider using clustering techniques like K-Means.")
    else:
        print("Dataset size is small. Clustering might not be required.")

    # 7. Hierarchy Detection
    print("\n📂 Basic Hierarchy Detection (Relationships between Columns):")
    category_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(category_cols) >= 2:
        print(f"Columns '{category_cols[0]}' and '{category_cols[1]}' might form hierarchical relationships.")
    else:
        print("Not enough categorical columns to detect hierarchy.")

    print("\n✅ Data Analysis Completed.\n")

# Example Usage
if __name__ == "__main__":
    # Replace 'dataset.csv' with your input CSV filename
    filename = "goodreads.csv"
    try:
        df = pd.read_csv(filename)
        analyze_data(df)
    except Exception as e:
        print("Error loading dataset:", e)

# Attempt to read with a fallback encoding
try:
    df = pd.read_csv("goodreads.csv", encoding="utf-8")
except UnicodeDecodeError:
    print("UTF-8 failed. Trying ISO-8859-1...")
    df = pd.read_csv("goodreads.csv", encoding="ISO-8859-1")  # Fallback encoding
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

print("Dataset loaded successfully.")

import chardet

with open("goodreads.csv", "rb") as f:
    result = chardet.detect(f.read())
    print("Detected Encoding:", result['encoding'])


