import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np
import re
from abbreviations import abbreviation_dict
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the datasets
train_data = pd.read_csv('data/ktas_train.csv', on_bad_lines='skip')
train_data.drop(columns=['Patients number per hour', 'Diagnosis in ED'], inplace=True)
val_data = pd.read_csv('data/ktas_val.csv', on_bad_lines='skip')
val_data.drop(columns=['Patients number per hour', 'Diagnosis in ED'], inplace=True)

# Initialize BERT tokenizer
tokenizer = AutoTokenizer.from_pretrained('nlpie/tiny-biobert')

def tokenize_texts(texts, tokenizer, max_len=6):
    input_ids = []
    attention_masks = []
    for text in texts:
        encoding = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        input_ids.append(encoding['input_ids'])
        attention_masks.append(encoding['attention_mask'])
    
    return torch.cat(input_ids), torch.cat(attention_masks)

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

# Preprocess and tokenize text data
train_data['Chief_complain'] = train_data['Chief_complain'].apply(clean_text)
val_data['Chief_complain'] = val_data['Chief_complain'].apply(clean_text)

train_input_ids, train_attention_masks = tokenize_texts(train_data['Chief_complain'], tokenizer)
val_input_ids, val_attention_masks = tokenize_texts(val_data['Chief_complain'], tokenizer)

# Normalize numerical data
scaler = StandardScaler()
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'NRS_pain', 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
val_data[numerical_columns] = scaler.transform(val_data[numerical_columns])

X_numerical_train = torch.tensor(train_data[numerical_columns].values, dtype=torch.float32)
X_numerical_val = torch.tensor(val_data[numerical_columns].values, dtype=torch.float32)

# Encode target labels
label_encoder = LabelEncoder()
y_train = torch.tensor(label_encoder.fit_transform(train_data['KTAS_expert'].values), dtype=torch.long)
y_val = torch.tensor(label_encoder.transform(val_data['KTAS_expert'].values), dtype=torch.long)

# Extract BERT embeddings
def extract_bert_embeddings(input_ids, attention_masks, model, device):
    embeddings = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(input_ids), 32):  # Batch size 32 for efficiency
            input_batch = input_ids[i:i+32].to(device)
            attention_batch = attention_masks[i:i+32].to(device)
            output = model(input_batch, attention_mask=attention_batch)
            batch_embeddings = output.last_hidden_state.mean(dim=1).cpu().numpy()  # Mean pooling
            embeddings.append(batch_embeddings)
    return np.vstack(embeddings)

bert_model = AutoModel.from_pretrained('nlpie/tiny-biobert').to(device)
train_bert_embeddings = extract_bert_embeddings(train_input_ids, train_attention_masks, bert_model, device)

# Combine BERT embeddings and numerical data
X_train_combined = np.concatenate((train_bert_embeddings, X_numerical_train.numpy()), axis=1)

# Apply SMOTE to combined data and labels
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train_combined, y_train.numpy())

# Find how many samples were added by SMOTE
n_synthetic = len(X_resampled) - len(train_data)

# Duplicate the tokenized data for synthetic samples
train_input_ids_resampled = torch.cat([train_input_ids, train_input_ids[:n_synthetic]], dim=0)
train_attention_masks_resampled = torch.cat([train_attention_masks, train_attention_masks[:n_synthetic]], dim=0)

# Convert the oversampled data back to PyTorch tensors
X_train_resampled_bert = torch.tensor(X_resampled[:, :train_bert_embeddings.shape[1]], dtype=torch.float32)
X_train_resampled_numerical = torch.tensor(X_resampled[:, train_bert_embeddings.shape[1]:], dtype=torch.float32)
y_train_resampled = torch.tensor(y_resampled, dtype=torch.long)

# Compute class weights
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train.numpy()),
    y=y_train.numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)

class MedicalRecordsDataset(Dataset):
    def __init__(self, input_ids, attention_masks, numerical_data, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.numerical_data = numerical_data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_masks[idx], self.numerical_data[idx], self.labels[idx])

# Create PyTorch datasets and dataloaders
train_dataset = MedicalRecordsDataset(train_input_ids_resampled, train_attention_masks_resampled, X_train_resampled_numerical, y_train_resampled)
val_dataset = MedicalRecordsDataset(val_input_ids, val_attention_masks, X_numerical_val, y_val)

train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

# Displaying raw examples
print(train_data[:5])

# Display samples from the dataset
print("Displaying some samples from the dataset:")
for i in range(5):  # Show 5 samples
    sample = train_dataset[i]
    print(f"Sample {i}:")
    print(f"Input IDs: {sample[0]}")
    print(f"Attention Masks: {sample[1]}")
    print(f"Numerical Data: {sample[2]}")
    print(f"Label: {sample[3]}")
    print()

# Displaying raw examples
print(val_data[:5])

# Display samples from the dataset
print("Displaying some samples from the dataset:")
for i in range(5):  # Show 5 samples
    sample = val_dataset[i]
    print(f"Sample {i}:")
    print(f"Input IDs: {sample[0]}")
    print(f"Attention Masks: {sample[1]}")
    print(f"Numerical Data: {sample[2]}")
    print(f"Label: {sample[3]}")
    print()

class MedicalRecordClassifier(nn.Module):
    def __init__(self, bert_model_name='nlpie/tiny-biobert', num_classes=5):
        super(MedicalRecordClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.hidden_size = self.bert.config.hidden_size
        
        # Define the linear layers
        self.fc1 = nn.Linear(self.hidden_size + len(numerical_columns), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
        # Define dropout and batch normalization
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.2)
        self.batch_norm = nn.BatchNorm1d(self.hidden_size + len(numerical_columns))

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, nonlinearity='relu')  # He initialization
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)  # BatchNorm scales are set to 1
                init.zeros_(m.bias)  # BatchNorm biases are set to 0

    def forward(self, input_ids, attention_mask, numerical_data):
        bert_output = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state.mean(dim=1)
        combined_features = torch.cat((bert_output, numerical_data), dim=1)
        
        # Apply BatchNorm
        combined_features = self.batch_norm(combined_features)
        
        combined_features = self.dropout1(nn.ReLU()(self.fc1(combined_features)))
        combined_features = self.dropout2(nn.ReLU()(self.fc2(combined_features)))
        output = self.fc3(combined_features)
        
        return output

# Instantiate the model
model = MedicalRecordClassifier()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

def train_model(model, train_dataloader, val_dataloader, epochs=100):
    history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': [], 'best_val_accuracy': 0.0}
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in train_dataloader:
            input_ids, attention_mask, numerical_data, labels = batch
            input_ids, attention_mask, numerical_data, labels = (
                input_ids.to(device),
                attention_mask.to(device),
                numerical_data.to(device),
                labels.to(device)
            )

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, numerical_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        epoch_accuracy = correct_predictions / total_samples

        # Validate
        model.eval()
        val_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        with torch.no_grad():
            for batch in val_dataloader:
                input_ids, attention_mask, numerical_data, labels = batch
                input_ids, attention_mask, numerical_data, labels = (
                    input_ids.to(device),
                    attention_mask.to(device),
                    numerical_data.to(device),
                    labels.to(device)
                )

                outputs = model(input_ids, attention_mask, numerical_data)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                correct_predictions += (predicted == labels).sum().item()
                total_samples += labels.size(0)

        val_loss /= len(val_dataloader.dataset)
        val_accuracy = correct_predictions / total_samples

        history['accuracy'].append(epoch_accuracy)
        history['val_accuracy'].append(val_accuracy)
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)

        print(f'Epoch {epoch+1}/{epochs}')
        print(f'Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

        # Track the best model
        if val_accuracy >= history['best_val_accuracy']:
            history['best_val_accuracy'] = val_accuracy
            torch.save(model.state_dict(), f"bert_model_{history['best_val_accuracy']:.4f}.pth")

    return history

# Train the model
history = train_model(model, train_dataloader, val_dataloader)

# Reload the best model and evaluate metrics
model.load_state_dict(torch.load(f"bert_model_{history['best_val_accuracy']:.4f}.pth"))
model.eval()
all_labels = []
all_preds = []

with torch.no_grad():
    for batch in val_dataloader:
        input_ids, attention_mask, numerical_data, labels = batch
        input_ids, attention_mask, numerical_data, labels = (
            input_ids.to(device),
            attention_mask.to(device),
            numerical_data.to(device),
            labels.to(device)
        )

        outputs = model(input_ids, attention_mask, numerical_data)
        _, predicted = torch.max(outputs, 1)
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

# Print classification report
class_names = [str(label) for label in label_encoder.classes_]
print("True labels: ", all_labels)
print("Predicted labels: ", all_preds)
print("Class names: ", class_names)
print(classification_report(all_labels, all_preds, target_names=class_names))

# Plot confusion matrix
cm = confusion_matrix(all_labels, all_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Plot accuracy and loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.show()

# https://huggingface.co/nlpie/tiny-biobert