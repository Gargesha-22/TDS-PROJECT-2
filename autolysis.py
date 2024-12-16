# -*- coding: utf-8 -*-
"""autolysis

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1ouGL_F94txC4H5r7_-2WyJiRHXhmVR1l
"""

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
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import requests
import json
import openai
import chardet

# Function to read the CSV file with proper encoding handling
def read_data_with_fallback_encoding(file_path):
    try:
        # First try with ISO-8859-1 encoding
        df = pd.read_csv(file_path, encoding='ISO-8859-1')
        print("File read successfully with ISO-8859-1 encoding.")
        return df
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None

# Data Analysis Function
def analyze_data(df):
    print("📝 Basic Dataset Information:")
    if df is None:
        print("Error: DataFrame is empty.")
        return None, None, None

    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")
    print(df.info())

    # 2. Summary Statistics for Numeric Columns
    print("\n📊 Summary Statistics (Numeric Columns):")
    numeric_df = df.select_dtypes(include=[np.number])  # Only numeric columns
    if numeric_df.empty:
        print("No numeric columns available.")
    else:
        print(numeric_df.describe())

    # 3. Missing Values
    print("\n🔍 Missing Values per Column:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # 4. Correlation Matrix for Numeric Columns
    print("\n📈 Correlation Matrix (Numeric Columns):")
    if not numeric_df.empty:
        correlation_matrix = numeric_df.corr()  # Only numeric columns
        print(correlation_matrix)
    else:
        correlation_matrix = pd.DataFrame()

    # Return the results
    return numeric_df.describe() if not numeric_df.empty else pd.DataFrame(), missing_values, correlation_matrix

# Function to generate the visualizations (one of each type)
def generate_visualizations(df):
    # 1. Missing Values Visualization
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if not missing_values.empty:
        plt.figure(figsize=(8, 6))
        missing_values.plot(kind="bar", color="red")
        plt.title("Missing Values per Column")
        plt.xlabel("Columns")
        plt.ylabel("Count of Missing Values")
        plt.savefig("missing_values.png")
        plt.close()
        print("✅ Missing values visualization saved.")
    else:
        print("No missing values found. Skipping visualization.")

    # 2. Correlation Matrix Heatmap
    numeric_cols = df.select_dtypes(include=["int64"])
    if not numeric_cols.empty:
        plt.figure(figsize=(10, 8))
        correlation_matrix = numeric_cols.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix Heatmap")
        plt.savefig("correlation_heatmap.png")
        plt.close()
        print("✅ Correlation matrix heatmap saved.")

    # 3. Bar Chart for the first categorical column (if any)
    category_cols = df.select_dtypes(include=["object", "category"]).columns
    if category_cols.size > 0:
        col = category_cols[0]  # Select the first categorical column
        plt.figure(figsize=(8, 6))
        df[col].value_counts().plot(kind="bar", color="orange")
        plt.title(f"Bar Chart for {col}")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.savefig(f"barchart_{col}.png")
        plt.close()
        print(f"✅ Bar chart for {col} saved.")

    print("\n🎉 All visualizations have been saved in the current directory.")

# Example usage
if __name__ == "__main__":
    # Example file path (replace with your actual file path)
    file_path = "/content/media.csv"  # Replace with your actual file path

    # Read data
    df = pd.read_csv(file_path, encoding="ISO-8859-1")

    # Generate the visualizations
    generate_visualizations(df)

def create_readme(df, summary_stats, missing_values, correlation_matrix, output_dir="."):
    # Path for the README file
    readme_file = os.path.join(output_dir, 'README.md')

    try:
        with open(readme_file, 'w') as f:
            # Introduction Section
            f.write("# Automated Data Analysis Report\n\n")
            f.write("## Overview\n")
            f.write("This script performs an automated analysis of the given dataset. It includes basic "
                    "information, summary statistics, missing values detection, correlation matrix, "
                    "outlier detection, and visualizations.\n\n")

            # Dataset Information Section
            f.write("## Dataset Information\n")
            f.write(f"- Shape: {df.shape}\n")
            f.write(f"- Columns: {', '.join(df.columns)}\n\n")
            f.write("### Data Types:\n")
            f.write(f"{df.dtypes}\n\n")

            # Summary Statistics Section
            f.write("## Summary Statistics\n")
            f.write("### Basic Statistical Summary:\n")
            f.write("\n| Statistic    | Value |\n")
            f.write("|--------------|-------|\n")
            for column in summary_stats.columns:
                f.write(f"| {column} - Mean | {summary_stats.loc['mean', column]:.2f} |\n")
                f.write(f"| {column} - Std Dev | {summary_stats.loc['std', column]:.2f} |\n")
                f.write(f"| {column} - Min | {summary_stats.loc['min', column]:.2f} |\n")
                f.write(f"| {column} - 25th Percentile | {summary_stats.loc['25%', column]:.2f} |\n")
                f.write(f"| {column} - 50th Percentile (Median) | {summary_stats.loc['50%', column]:.2f} |\n")
                f.write(f"| {column} - 75th Percentile | {summary_stats.loc['75%', column]:.2f} |\n")
                f.write(f"| {column} - Max | {summary_stats.loc['max', column]:.2f} |\n")
                f.write("|--------------|-------|\n")

            # Missing Values Section
            f.write("## Missing Values\n")
            f.write("The following columns contain missing values, with their respective counts:\n")
            f.write("\n| Column       | Missing Values Count |\n")
            f.write("|--------------|----------------------|\n")
            for column, count in missing_values.items():
                f.write(f"| {column} | {count} |\n")
            f.write("\n")

            # Correlation Matrix Section
            f.write("## Correlation Matrix\n")
            f.write("Below is the correlation matrix of numerical features, indicating relationships between different variables:\n\n")
            f.write("![Correlation Matrix](correlation_heatmap.png)\n\n")

            # Distribution Plot Section
            f.write("## Distribution of Data\n")
            if os.path.exists("distribution_.png"):
                f.write("Below is the distribution plot of the first numerical column in the dataset:\n\n")
                f.write("![Distribution](distribution_.png)\n\n")
            else:
                f.write("No distribution plot generated.\n\n")

            # Conclusion Section
            f.write("## Conclusion\n")
            f.write("The analysis has provided insights into the dataset, including summary statistics, outlier detection, and correlations between key variables.\n")
            f.write("The generated visualizations and statistical insights can help in understanding the patterns and relationships in the data.\n\n")

            print(f"README file created: {readme_file}")
            return readme_file
    except Exception as e:
        print(f"Error writing to README.md: {e}")
        return None

if __name__ == "__main__":
    # Example file path (replace with your actual file path)
    file_path = "/content/media.csv"  # Replace with your actual file path

    # Read data with fallback encoding
    df = read_data_with_fallback_encoding(file_path)

    # Perform the data analysis
    summary_stats, missing_values, corr_matrix = analyze_data(df)

    # Generate visualizations
    generate_visualizations(df)

    # Create the README file with the analysis and the story
    readme_file = create_readme(df, summary_stats, missing_values, corr_matrix, output_dir=".")

    if readme_file:
        print(f"Analysis complete! Results saved in '{readme_file}'")

def generate_detailed_story(df, output_dir=".", images=None, api_token=None):
    """
    Generates a detailed story based on the dataset analysis.

    Parameters:
    - df: Pandas DataFrame with the dataset.
    - output_dir: Directory to save the README.md and images.
    - images: List of image filenames to include in the narrative (optional).
    - api_token: The AI Proxy token for OpenAI API.

    Returns:
    - The generated story as a string.
    """

    # Set the AI Proxy token
    openai.api_key = api_token or os.getenv("AIPROXY_TOKEN")

    # Prepare the summary data to send to the LLM
    dataset_info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "data_types": str(df.dtypes),
        "summary_statistics": df.describe().to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "correlation_matrix": df.select_dtypes(include=[np.number]).corr().to_dict()  # Only numeric columns
    }

    # Step 2: Create the prompt for the LLM
    prompt = f"""
    Write a detailed story about the analysis of the following dataset. The story should be coherent,
    explaining the dataset, analysis steps, insights gained, and implications. Include insights about
    the dataset's structure, summary statistics, missing data, relationships between variables, outliers,
    clustering suggestions, and potential hierarchical relationships between categorical columns.

    Here is the dataset information:
    Dataset Shape: {dataset_info['shape']}
    Columns: {', '.join(dataset_info['columns'])}
    Data Types: {dataset_info['data_types']}

    Summary Statistics:
    {json.dumps(dataset_info['summary_statistics'], indent=2)}

    Missing Values per Column:
    {json.dumps(dataset_info['missing_values'], indent=2)}

    Correlation Matrix (Numerical Columns):
    {json.dumps(dataset_info['correlation_matrix'], indent=2)}

    If there are any outliers, explain their significance. If there are relationships between categorical
    columns, describe the potential hierarchical structures. Based on the dataset size, suggest whether
    clustering techniques like K-Means would be beneficial.

    You may include visualizations to support your narrative. Refer to the images (PNG files) below:
    """

    # Step 3: If there are images (visualizations), include them in the prompt
    image_references = []
    if images:
        for image in images:
            image_path = os.path.join(output_dir, image)
            image_references.append(f"![{image}]({image_path})")

    # Include the image references in the prompt
    prompt += "\n".join(image_references) + "\n"

    # Step 4: Request the detailed story from the LLM via AI Proxy (OpenAI API)
    try:
        # Request the LLM to generate the story using the new API interface
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0.7,
            n=1,
            stop=["\n"]
        )

        # Extract the story from the response
        story = response['choices'][0]['message']['content'].strip()

        # Save the generated story to a file
        story_file = os.path.join(output_dir, "analysis_story.md")
        with open(story_file, "w") as f:
            f.write(story)

        print(f"Story generated successfully and saved to {story_file}")

        return story

    except Exception as e:
        print(f"Error generating story: {e}")
        return None

# Perform the analysis and generate the story
analyze_data(df)
story = generate_detailed_story(df, output_dir=".", api_token="eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjEwMDE3NTFAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.Jstz7rAnOcYgOdYoRDJCZAuJltFnACH_0yBKYTNdJ0Y")