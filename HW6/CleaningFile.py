import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv("Credit Score Classification Dataset.csv")

# Make a copy of the dataframe
df_cleaned = df.copy()

# Map binary features
df_cleaned['Gender'] = df_cleaned['Gender'].map({'Male': 0, 'Female': 1})
df_cleaned['Marital Status'] = df_cleaned['Marital Status'].map({'Single': 0, 'Married': 1})
df_cleaned['Home Ownership'] = df_cleaned['Home Ownership'].map({'Rented': 0, 'Owned': 1})
df_cleaned['Credit Score'] = df_cleaned['Credit Score'].map({'Low': 0, 'Average': 1, 'High': 2})

# Drop the 'Education' column
df_cleaned = df_cleaned.drop(columns=['Education'])

# Check the cleaned data
print(df_cleaned)
df_cleaned.to_csv("cleanedCreditScore.csv", index=False)
