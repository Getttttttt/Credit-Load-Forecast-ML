# Merge all dataframes on the 'id' column
from functools import reduce
import pandas as pd

# Load the CSV files
file_paths = [
    "./Statistics/data/submission_0.9068.csv",
    "./Statistics/data/submission_0.9075.csv",
    "./Statistics/data/submission_0.9078.csv",
    "./Statistics/data/submission_0.90405.csv",
    "./Statistics/data/submission_0.90811.csv",
    "./Statistics/data/submission_0.90821.csv"
]

# Dictionary to store the dataframes
dfs = {}

# Read each file and store the dataframe with a key as the file name without extension
for path in file_paths:
    df_name = path.split("/")[-1].split(".")[0]
    dfs[df_name] = pd.read_csv(path)

# Display the first few rows of each dataframe to check their structure
{key: df.head() for key, df in dfs.items()}

# Correct naming issue and display each DataFrame's structure
dfs = {}
for path in file_paths:
    df_name = path.split("/")[-1].split(".csv")[0]  # Correct name extraction by removing '.csv'
    dfs[df_name] = pd.read_csv(path)

# Display the first few rows of each dataframe to check their structure
{key: df.head() for key, df in dfs.items()}


# Convert id to integer for accurate merging
for df in dfs.values():
    df['id'] = df['id'].astype(int)

# Perform the merging
merged_df = reduce(lambda left, right: pd.merge(left, right, on='id', suffixes=('', '_right')), dfs.values())

# Rename the columns to use the original file names
column_names = ['id'] + [name for name in dfs.keys()]
merged_df.columns = column_names

# Display the first few rows of the merged dataframe
merged_df.head()
