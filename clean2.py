import pandas as pd

# Read the CSV in chunks
data_iterator = pd.read_csv('Hospital Triage and Patient History.csv', on_bad_lines='skip', delimiter=',', encoding='utf-8', chunksize=1000)

# Get the first chunk
first_chunk = next(data_iterator)

# Print the head of the first chunk
print(first_chunk.head())
