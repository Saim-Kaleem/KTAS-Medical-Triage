import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from abbreviations import abbreviation_dict
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from scipy import interp
from itertools import cycle

# Load and preprocess the dataset
data = pd.read_csv('data/data_cleaned2.csv', on_bad_lines='skip')

# Required columns
text_columns = ['Chief_complain']
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
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

data['Chief_complain'] = data['Chief_complain'].apply(clean_text)

# Tokenize the text data
data['tokens'] = data['Chief_complain'].apply(lambda x: x.split())

# Load pretrained embeddings
biowordvec_model = KeyedVectors.load_word2vec_format('embeddings/bio_embedding_extrinsic', binary=True)
embedding_dim = biowordvec_model.vector_size

# Map tokens to embeddings
def get_embeddings(tokens, model, vector_size):
    embeddings = [model[word] if word in model else np.zeros(vector_size) for word in tokens]
    return np.array(embeddings)

data['embeddings'] = data['tokens'].apply(lambda x: get_embeddings(x, biowordvec_model, embedding_dim))

# Convert KTAS_expert to numeric labels
label_encoder = LabelEncoder()
data['KTAS_expert'] = label_encoder.fit_transform(data['KTAS_expert'])

# Normalize numerical data
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Prepare the inputs
max_length = max(data['tokens'].apply(len))
average_length = np.mean(data['tokens'].apply(len))
print(f"Maximum sequence length: {max_length}")
print(f"Average sequence length: {average_length}")

# Padding sequences for PyTorch
def pad_sequences(sequences, maxlen, embedding_dim):
    padded_sequences = np.zeros((len(sequences), maxlen, embedding_dim), dtype=np.float32)
    for i, seq in enumerate(sequences):
        if len(seq) > 0:
            # Ensure the sequence length does not exceed the max length
            length = min(len(seq), maxlen)
            padded_sequences[i, :length] = seq[:length]
        else:
            # Handle empty sequences with zeros (already zeroed out by np.zeros initialization)
            padded_sequences[i] = np.zeros((maxlen, embedding_dim))
    return padded_sequences

X_text = pad_sequences(data['embeddings'].tolist(), max_length, embedding_dim)
X_numerical = data[numerical_columns].values
y = data['KTAS_expert'].values

print(f"X_text shape: {X_text.shape}")
print(f"X_numerical shape: {X_numerical.shape}")
print(f"y shape: {y.shape}")

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, X_text, X_numerical, y):
        self.X_text = torch.tensor(X_text, dtype=torch.float32)
        self.X_numerical = torch.tensor(X_numerical, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X_text[idx], self.X_numerical[idx], self.y[idx]

# Define the PyTorch model architecture

class KTASModel(nn.Module):
    def __init__(self, text_input_shape, numerical_input_shape, output_size):
        super(KTASModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=text_input_shape[1], out_channels=128, kernel_size=3)
        self.pooling = nn.AdaptiveMaxPool1d(1)
        self.fc_query = nn.Linear(128, 128)
        self.fc_key = nn.Linear(128, 128)
        self.fc_value = nn.Linear(128, 128)
        self.lstm = nn.LSTM(input_size=text_input_shape[1], hidden_size=128, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc1 = nn.Linear(128 * 2 + numerical_input_shape[0], 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)

        # Apply Kaiming He initialization
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize Conv1d layer
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_in', nonlinearity='relu')
        if self.conv1d.bias is not None:
            nn.init.zeros_(self.conv1d.bias)

        # Initialize Linear layers
        nn.init.kaiming_normal_(self.fc_query.weight, mode='fan_in', nonlinearity='relu')
        if self.fc_query.bias is not None:
            nn.init.zeros_(self.fc_query.bias)
        
        nn.init.kaiming_normal_(self.fc_key.weight, mode='fan_in', nonlinearity='relu')
        if self.fc_key.bias is not None:
            nn.init.zeros_(self.fc_key.bias)
        
        nn.init.kaiming_normal_(self.fc_value.weight, mode='fan_in', nonlinearity='relu')
        if self.fc_value.bias is not None:
            nn.init.zeros_(self.fc_value.bias)
        
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
        
        nn.init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity='relu')
        if self.fc3.bias is not None:
            nn.init.zeros_(self.fc3.bias)

    def attention(self, x):
        query = self.fc_query(x)
        key = self.fc_key(x)
        value = self.fc_value(x)
        attention_scores = torch.matmul(query, key.transpose(-2, -1))
        attention_scores = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_scores, value)
        return attention_output

    def forward(self, text_inputs, numerical_inputs):
        x = text_inputs.permute(0, 2, 1)  # (batch_size, embedding_dim, seq_len)
        x = F.relu(self.conv1d(x))
        x = self.pooling(x).squeeze(-1)
        attention_output = self.attention(x)
        x = x + attention_output  # Skip connection
        x = nn.BatchNorm1d(x.size(1)).to(x.device)(x)
        x, _ = self.lstm(text_inputs)
        x = x[:, -1, :]  # Last output for each sequence
        combined_inputs = torch.cat((x, numerical_inputs), dim=1)
        combined_inputs = nn.BatchNorm1d(combined_inputs.size(1)).to(combined_inputs.device)(combined_inputs)
        x = F.relu(self.fc1(combined_inputs))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Stratified k-fold validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
all_histories = []
best_val_accuracies = []

# For ROC curves
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for train_index, val_index in kfold.split(X_text, y):
    print(f"Training fold {fold_no}...")

    # Split the data
    X_text_train, X_text_val = X_text[train_index], X_text[val_index]
    X_numerical_train, X_numerical_val = X_numerical[train_index], X_numerical[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    # Reshape X_text_train to a 2D array for SMOTE
    X_text_train_flat = X_text_train.reshape(X_text_train.shape[0], -1)
    X_train_combined = np.hstack((X_text_train_flat, X_numerical_train))
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_combined, y_train)
    
    # Reshape X_train_resampled back to original shape for text and numerical data
    X_text_train_resampled = X_train_resampled[:, :X_text_train_flat.shape[1]].reshape(-1, max_length, embedding_dim)
    X_numerical_train_resampled = X_train_resampled[:, X_text_train_flat.shape[1]:]

    # Create Datasets and DataLoaders using resampled data
    train_dataset = CustomDataset(X_text_train_resampled, X_numerical_train_resampled, y_train_resampled)
    val_dataset = CustomDataset(X_text_val, X_numerical_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    # Initialize model, loss, and optimizer
    model = KTASModel(text_input_shape=(max_length, embedding_dim), numerical_input_shape=(len(numerical_columns),), output_size=5)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    best_val_accuracy = 0.0
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Training phase
        for text_inputs, numerical_inputs, labels in train_loader:
            text_inputs, numerical_inputs, labels = text_inputs.to(device), numerical_inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(text_inputs, numerical_inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * text_inputs.size(0)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_accuracy = correct / total
        train_loss = running_loss / total

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for text_inputs, numerical_inputs, labels in val_loader:
                text_inputs, numerical_inputs, labels = text_inputs.to(device), numerical_inputs.to(device), labels.to(device)
                outputs = model(text_inputs, numerical_inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * text_inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = val_correct / val_total
        val_loss /= val_total

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        
        # Track the best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), f"ktas_model_fold{fold_no}_{best_val_accuracy:.4f}.pth")

    all_histories.append({'val_accuracy': best_val_accuracy})
    best_val_accuracies.append(best_val_accuracy)
    
    # Reload the best model and evaluate metrics
    model.load_state_dict(torch.load(f"ktas_model_fold{fold_no}_{best_val_accuracy:.4f}.pth"))
    model.eval()
    val_outputs = []
    val_labels = []
    with torch.no_grad():
        for text_inputs, numerical_inputs, labels in val_loader:
            text_inputs, numerical_inputs, labels = text_inputs.to(device), numerical_inputs.to(device), labels.to(device)
            outputs = model(text_inputs, numerical_inputs)
            val_outputs.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    val_outputs = np.array(val_outputs)
    val_labels = np.array(val_labels)

    # Calculate ROC and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(5):
        fpr[i], tpr[i], _ = roc_curve(val_labels == i, val_outputs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        tprs.append(np.interp(mean_fpr, fpr[i], tpr[i]))
        tprs[-1][0] = 0.0
        aucs.append(roc_auc[i])
    
    fold_no += 1

# Plot ROC curve for each fold
plt.figure(figsize=(10, 7))
mean_tpr = np.mean(tprs, axis=0)
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()