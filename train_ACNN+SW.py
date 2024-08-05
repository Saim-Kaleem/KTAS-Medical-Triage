import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dense, Dropout, Dot, Softmax, Multiply, Add, Concatenate, BatchNormalization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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

# Train a Word2Vec model
w2v_model = Word2Vec(sentences=data['tokens'], vector_size=50, window=3, min_count=1, workers=4)

# Create an embedding matrix
embedding_matrix = w2v_model.wv
embedding_dim = embedding_matrix.vector_size

# Map tokens to embeddings
def get_word2vec_embeddings(tokens, model, vector_size):
    embeddings = [model.wv[word] if word in model.wv else np.zeros(vector_size) for word in tokens]
    return np.array(embeddings)

data['embeddings'] = data['tokens'].apply(lambda x: get_word2vec_embeddings(x, w2v_model, embedding_dim))

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
X_text = pad_sequences(data['embeddings'].apply(lambda x: [embedding for embedding in x]), maxlen=max_length, dtype='float32')
X_numerical = data[numerical_columns].values
y = data['KTAS_expert'].values

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# Apply SMOTE to balance the classes
smote = SMOTE()

# Define the model architecture
text_input_shape = (max_length, embedding_dim)
numerical_input_shape = (len(numerical_columns),)

text_inputs = Input(shape=text_input_shape, name='text_input')
numerical_inputs = Input(shape=numerical_input_shape, name='numerical_input')

# CNN-based n-gram encoder for text inputs
conv_layer = Conv1D(filters=128, kernel_size=3, activation='relu', kernel_initializer='he_normal')(text_inputs)
conv_layer = GlobalMaxPooling1D()(conv_layer)

# Attention mechanism using Dot product
query = Dense(128)(conv_layer)
key = Dense(128)(conv_layer)
value = Dense(128)(conv_layer)

attention_scores = Dot(axes=-1)([query, key])
attention_scores = Softmax()(attention_scores)
attention_output = Multiply()([attention_scores, value])

# Adding the attention output to the convolution output (skip connection)
attention_output = Add()([conv_layer, attention_output])

# Apply batch normalization to the attention output
attention_output = BatchNormalization()(attention_output)

# Concatenate text and numerical inputs
combined_inputs = Concatenate()([attention_output, numerical_inputs])

# Fully connected layers
fc_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(combined_inputs)
fc_layer = Dropout(0.3)(fc_layer)
fc_layer = Dense(32, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(fc_layer)
fc_layer = Dropout(0.2)(fc_layer)

# Output layer with softmax activation
output_layer = Dense(5, activation='softmax')(fc_layer)

# Build the model
model = Model(inputs=[text_inputs, numerical_inputs], outputs=output_layer)

# Set an optimizer and compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Custom training loop
epochs = 40
batch_size = 16
history = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    
    if epoch < 8:
        # Apply class weights
        current_history = model.fit([X_text, X_numerical], y, epochs=1, batch_size=batch_size, validation_split=0.2, shuffle=True, class_weight=class_weights_dict)
    else:
        # Apply SMOTE
        X_text_resampled, y_resampled = smote.fit_resample(X_text.reshape((X_text.shape[0], -1)), y)
        X_numerical_resampled, _ = smote.fit_resample(X_numerical, y)
        X_text_resampled = X_text_resampled.reshape((-1, max_length, embedding_dim))
        current_history = model.fit([X_text_resampled, X_numerical_resampled], y_resampled, epochs=1, batch_size=batch_size, validation_split=0.2, shuffle=True)
    
    # Store the history
    for key in history.keys():
        history[key].extend(current_history.history[key])

# Save the model
model.save('ktas_model.keras')

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

# Convert the class labels to string format
class_names = [str(label) for label in label_encoder.classes_]

# Evaluate the model on the entire dataset to get precision, recall, and f1-score
y_pred = model.predict([X_text, X_numerical])
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y, y_pred_classes, target_names=class_names))