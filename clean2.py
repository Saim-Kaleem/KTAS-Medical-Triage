import pandas as pd

data = pd.read_csv('data/data_utf8.csv', on_bad_lines='skip', delimiter=';')

# Drop columns that are not needed
data = data.drop(columns=['Group', 'KTAS_RN', 'Disposition', 'Error_group', 'KTAS duration_min', 'mistriage', 'Length of stay_min'])

# Decode columns with number values
data['Sex'] = data['Sex'].replace({1: 0, 2: 1})
data['Arrival mode'] = data['Arrival mode'].replace({1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6})
data['Injury'] = data['Injury'].replace({1: 0, 2: 1})
data['Mental'] = data['Mental'].replace({1: 0, 2: 1, 3: 2, 4: 3})
data['Pain'] = data['Pain'].replace({2: 0})

# Convert relevant columns to numeric values
data['SBP'] = pd.to_numeric(data['SBP'], errors='coerce')
data['DBP'] = pd.to_numeric(data['DBP'], errors='coerce')
data['HR'] = pd.to_numeric(data['HR'], errors='coerce')
data['RR'] = pd.to_numeric(data['RR'], errors='coerce')
data['BT'] = pd.to_numeric(data['BT'], errors='coerce')
data['Saturation'] = pd.to_numeric(data['Saturation'], errors='coerce')
data['NRS_pain'] = pd.to_numeric(data['NRS_pain'], errors='coerce')

# Replace NaN values with "normal" values for each category
data['SBP'].fillna(110, inplace=True)  # Normal SBP
data['DBP'].fillna(70, inplace=True)   # Normal DBP
data['HR'].fillna(75, inplace=True)    # Normal HR
data['RR'].fillna(16, inplace=True)    # Normal RR
data['BT'].fillna(37, inplace=True)    # Normal BT
data['Saturation'].fillna(98, inplace=True)  # Normal Saturation
data['NRS_pain'].fillna(4, inplace=True)  # No pain

print(data.head(10))
data.to_csv('data/data_cleaned2.csv', index=False)