import pandas as pd
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, TensorDataset
from gensim.models import Word2Vec
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from abbreviations import abbreviation_dict

# Load and preprocess the dataset
data = pd.read_csv('data_cleaned.csv', on_bad_lines='skip')

# Required columns
text_columns = ['Chief_complain']
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'BP', 'HR', 'RR', 'BT', 'Saturation']
label_column = 'KTAS_expert'

# Shuffle rows in the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Clean the text data
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        text = text.replace(',', '').replace('?', '')  # Remove commas and question marks
        words = text.split()
        words = [abbreviation_dict.get(word, word) for word in words]
        text = ' '.join(words)
        text = text.lower()  # Convert to lowercase
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
        text = re.sub(r'\d+', '', text)  # Remove digits
    return text

data['Chief_complain'] = data['Chief_complain'].apply(clean_text)

# Tokenize the text data
data['tokens'] = data['Chief_complain'].apply(lambda x: x.split())

# Train a Word2Vec model
w2v_model = Word2Vec(sentences=data['tokens'], vector_size=50, window=3, min_count=1, workers=4)

# Create an embedding matrix
embedding_matrix = w2v_model.wv
embedding_dim = embedding_matrix.vector_size

# Map tokens to embeddings
def get_word2vec_embeddings(tokens, model, vector_size):
    if not tokens:
        return np.zeros((1, vector_size))
    embeddings = [model.wv[word] if word in model.wv else np.zeros(vector_size) for word in tokens]
    return np.array(embeddings)

data['embeddings'] = data['tokens'].apply(lambda x: get_word2vec_embeddings(x, w2v_model, embedding_dim))

# Convert KTAS_expert to numeric labels
label_encoder = LabelEncoder()
data['KTAS_expert'] = label_encoder.fit_transform(data['KTAS_expert'])

# Normalize numerical data
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# Convert embeddings into a list of tensors
embeddings_list = [torch.tensor(embedding, dtype=torch.float32) for embedding in data['embeddings']]

# Pad sequences to the maximum length in the dataset
X_text = pad_sequence(embeddings_list, batch_first=True)

# Ensure numerical data and labels are also tensors
X_numerical = torch.tensor(data[numerical_columns].values, dtype=torch.float32)
y = torch.tensor(data['KTAS_expert'].values, dtype=torch.long)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y.numpy()), y=y.numpy())
class_weights = torch.tensor(class_weights, dtype=torch.float32)

# Apply SMOTE to balance the classes after epoch 8
smote = SMOTE()

# Define the model architecture in PyTorch
class ACNNModel(nn.Module):
    def __init__(self, text_input_shape, numerical_input_shape, num_classes=5):
        super(ACNNModel, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=text_input_shape[1], out_channels=128, kernel_size=3, padding=1)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.query_fc = nn.Linear(128, 128)
        self.key_fc = nn.Linear(128, 128)
        self.value_fc = nn.Linear(128, 128)
        self.softmax = nn.Softmax(dim=-1)
        self.batch_norm = nn.BatchNorm1d(128)
        self.fc1 = nn.Linear(128 + numerical_input_shape[0], 64)
        self.fc2 = nn.Linear(64, 32)
        self.output = nn.Linear(32, num_classes)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.l2_reg = nn.Parameter(torch.tensor(0.01, requires_grad=True))

    def forward(self, text_input, numerical_input):
        x = self.conv1d(text_input.permute(0, 2, 1))
        x = self.global_max_pool(x).squeeze(-1)
        query = self.query_fc(x)
        key = self.key_fc(x)
        value = self.value_fc(x)
        attention_scores = self.softmax(torch.matmul(query, key.transpose(-2, -1)))
        attention_output = torch.matmul(attention_scores, value)
        x = x + attention_output
        x = self.batch_norm(x)
        combined_input = torch.cat([x, numerical_input], dim=1)
        x = self.dropout1(torch.relu(self.fc1(combined_input)))
        x = self.dropout2(torch.relu(self.fc2(x)))
        output = self.output(x)
        return output

# Initialize the model
text_input_shape = (X_text.shape[1], embedding_dim)
numerical_input_shape = (X_numerical.shape[1],)
model = ACNNModel(text_input_shape, numerical_input_shape)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Custom training loop
epochs = 40
batch_size = 16
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

# Create a DataLoader for batching
dataset = TensorDataset(X_text, X_numerical, y)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for X_text_batch, X_numerical_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_text_batch, X_numerical_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = correct / total
    history['accuracy'].append(epoch_acc)
    history['loss'].append(epoch_loss)

    # Validation (using the same training set for simplicity)
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_text, X_numerical)
        val_loss = criterion(val_outputs, y)
        _, val_predicted = torch.max(val_outputs.data, 1)
        val_acc = (val_predicted == y).sum().item() / y.size(0)
    
    history['val_accuracy'].append(val_acc)
    history['val_loss'].append(val_loss.item())
    
    print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
    
    if epoch >= 8:
        # Apply SMOTE for oversampling
        X_text_flat = X_text.reshape((X_text.size(0), -1)).numpy()
        X_text_resampled, y_resampled = smote.fit_resample(X_text_flat, y.numpy())
        X_numerical_resampled, _ = smote.fit_resample(X_numerical.numpy(), y.numpy())
        X_text = torch.tensor(X_text_resampled, dtype=torch.float32).reshape((-1, text_input_shape[0], text_input_shape[1]))
        X_numerical = torch.tensor(X_numerical_resampled, dtype=torch.float32)
        y = torch.tensor(y_resampled, dtype=torch.long)
        dataset = TensorDataset(X_text, X_numerical, y)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Save the model
torch.save(model.state_dict(), 'ktas_model.pth')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()

# Evaluate the model on the entire dataset to get precision, recall, and f1-score
model.eval()
with torch.no_grad():
    y_pred = model(X_text, X_numerical)
    y_pred_classes = torch.argmax(y_pred, dim=1).numpy()

# Convert the class labels to string format
class_names = [str(label) for label in label_encoder.classes_]

# Print classification report
print(classification_report(y.numpy(), y_pred_classes, target_names=class_names))