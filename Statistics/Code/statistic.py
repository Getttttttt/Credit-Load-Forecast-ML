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

# Define the rule for the new 'isDefault' submission column
def determine_default(row):
    # Sort the values to easily determine the best (minimum) value
    sorted_values = sorted(row[1:])  # exclude 'id' column
    count_below_07 = sum(value < 0.07 for value in sorted_values)
    best = sorted_values[0]

    # Rule: All values below 0.07
    if all(value < 0.07 for value in sorted_values):
        return 0
    # Rule: 4 values below 0.07 including the best
    elif count_below_07 >= 4:
        return 0
    # Rule: Best below 0.1 and others below 0.2
    elif best < 0.1 and all(value < 0.2 for value in sorted_values):
        return best / 2
    # Rule: All values above 0.8
    elif all(value > 0.8 for value in sorted_values):
        return 1
    # Default: no change
    else:
        return best

# Apply the rules to generate the new 'isDefault' column
merged_df['isDefault'] = merged_df.apply(determine_default, axis=1)

# Extract the 'id' and new 'isDefault' columns to a new dataframe
final_submission_df = merged_df[['id', 'isDefault']]

# Display the first few rows of the new submission dataframe
final_submission_df.head()
