import pandas as pd
import matplotlib.pyplot as plt

# Load the data
data = pd.read_csv('data/data_cleaned2.csv', on_bad_lines='skip')
data = data.drop(columns=['Chief_complain', 'Diagnosis in ED'], axis=1)
print(data.head())

# Calculate the correlation matrix
correlation_matrix = data.corr()

# Plot the correlation matrix
plt.figure(figsize=(12, 10))
plt.matshow(correlation_matrix, fignum=1)
plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation='vertical')
plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
plt.colorbar()
plt.title('Correlation Matrix')
plt.show()

# Find the highest correlations with the target variable 'KTAS_expert'
print(correlation_matrix['KTAS_expert'].sort_values(ascending=False))