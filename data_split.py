import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt

# Load your KTAS dataset
data = pd.read_csv('data/data_cleaned2.csv', on_bad_lines='skip')

# Shuffle the data to randomize the order
# data = shuffle(data, random_state=1)

# Separate features and target label
X = data.drop(columns=['KTAS_expert'])
y = data['KTAS_expert']

# Split the dataset into training and validation sets with stratification
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Combine features and target for training and validation sets
train_data = pd.concat([X_train, y_train], axis=1)
val_data = pd.concat([X_val, y_val], axis=1)

# Display the distribution of KTAS labels in the training and validation sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Training Set')
train_data['KTAS_expert'].value_counts().plot(kind='bar')
plt.xlabel('KTAS Label')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
plt.title('Validation Set')
val_data['KTAS_expert'].value_counts().plot(kind='bar')
plt.xlabel('KTAS Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Merge the training and validation sets back into a single dataset
df = pd.concat([train_data, val_data])
df.to_csv('data/ktas_full.csv', index=False)

# Save the training and validation sets to CSV files
train_data.to_csv('data/ktas_train.csv', index=False)
val_data.to_csv('data/ktas_val.csv', index=False)

print("Data split completed. Training and validation sets saved as 'ktas_train.csv' and 'ktas_val.csv'.")