import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils import class_weight, shuffle
from imblearn.over_sampling import SMOTE
from abbreviations import abbreviation_dict
import matplotlib.pyplot as plt

# Load and preprocess the dataset
train_data = pd.read_csv('data/ktas_train.csv', on_bad_lines='skip')
val_data = pd.read_csv('data/ktas_val.csv', on_bad_lines='skip')

# Required columns
text_columns = ['Chief_complain']
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
label_column = 'KTAS_expert'

# Clean the text data
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        # Remove commas and question marks
        text = text.replace(',', '').replace('?', '')

        # Replace abbreviations
        words = text.split()  # Split text into words
        words = [abbreviation_dict.get(word, word) for word in words]
        text = ' '.join(words)  # Join words back into a sentence
        
        # Convert to lowercase and remove any other unwanted characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
    return text

train_data['Chief_complain'] = train_data['Chief_complain'].apply(clean_text)
val_data['Chief_complain'] = val_data['Chief_complain'].apply(clean_text)

# Convert KTAS_expert to numeric labels
label_encoder = LabelEncoder()
train_data['KTAS_expert'] = label_encoder.fit_transform(train_data['KTAS_expert'])
val_data['KTAS_expert'] = label_encoder.transform(val_data['KTAS_expert'])

# Normalize numerical data
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
val_data[numerical_columns] = scaler.transform(val_data[numerical_columns])

# TF-IDF Vectorizer for text data
tfidf_vectorizer = TfidfVectorizer(max_features=5000)

# Transform text data to TF-IDF features
X_text_train = tfidf_vectorizer.fit_transform(train_data['Chief_complain']).toarray()
X_text_val = tfidf_vectorizer.transform(val_data['Chief_complain']).toarray()

# Combine text and numerical data
X_train = np.hstack((X_text_train, train_data[numerical_columns].values))
X_val = np.hstack((X_text_val, val_data[numerical_columns].values))
y_train = train_data['KTAS_expert'].values
y_val = val_data['KTAS_expert'].values

# Apply SMOTE
smote = SMOTE()
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display the distribution of KTAS labels in the training and validation sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Training Set')
pd.Series(y_train_resampled).value_counts().plot(kind='bar')
plt.xlabel('KTAS Label')
plt.ylabel('Count')
plt.subplot(1, 2, 2)
plt.title('Validation Set')
pd.Series(y_val).value_counts().plot(kind='bar')
plt.xlabel('KTAS Label')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_resampled), y=y_train_resampled)
class_weights_dict = dict(enumerate(class_weights))

# Define the XGBoost model
model = XGBClassifier(
    objective='multi:softmax',
    num_class=5,
    eval_metric='mlogloss',
    scale_pos_weight=class_weights_dict,
    use_label_encoder=False
)

# Train the model
model.fit(X_train_resampled, y_train_resampled, eval_set=[(X_val, y_val)])

# Predict on validation set
y_pred = model.predict(X_val)

# Print classification report
class_names = [str(label) for label in label_encoder.classes_]
print(classification_report(y_val, y_pred, target_names=class_names))