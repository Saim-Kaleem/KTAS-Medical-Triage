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
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from abbreviations import abbreviation_dict

# Load the training and validation datasets
train_data = pd.read_csv('data/ktas_train.csv', on_bad_lines='skip')
val_data = pd.read_csv('data/ktas_val.csv', on_bad_lines='skip')

# Required columns
text_columns = ['Chief_complain']
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'NRS_pain' 'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation']
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

# Apply text cleaning
train_data['Chief_complain'] = train_data['Chief_complain'].apply(clean_text)
val_data['Chief_complain'] = val_data['Chief_complain'].apply(clean_text)

# Tokenize the text data
train_data['tokens'] = train_data['Chief_complain'].apply(lambda x: x.split())
val_data['tokens'] = val_data['Chief_complain'].apply(lambda x: x.split())

# Train a Word2Vec model using the training data
w2v_model = Word2Vec(sentences=train_data['tokens'], vector_size=50, window=3, min_count=1, workers=4)

# Create an embedding matrix
embedding_matrix = w2v_model.wv
embedding_dim = embedding_matrix.vector_size

# Map tokens to embeddings
def get_word2vec_embeddings(tokens, model, vector_size):
    embeddings = [model.wv[word] if word in model.wv else np.zeros(vector_size) for word in tokens]
    return np.array(embeddings)

train_data['embeddings'] = train_data['tokens'].apply(lambda x: get_word2vec_embeddings(x, w2v_model, embedding_dim))
val_data['embeddings'] = val_data['tokens'].apply(lambda x: get_word2vec_embeddings(x, w2v_model, embedding_dim))

# Convert KTAS_expert to numeric labels
label_encoder = LabelEncoder()
train_data['KTAS_expert'] = label_encoder.fit_transform(train_data['KTAS_expert'])
val_data['KTAS_expert'] = label_encoder.transform(val_data['KTAS_expert'])

# Normalize numerical data
scaler = StandardScaler()
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
val_data[numerical_columns] = scaler.transform(val_data[numerical_columns])

# Prepare the inputs
max_length = max(train_data['tokens'].apply(len))
average_length = np.mean(train_data['tokens'].apply(len))
print(f"Maximum sequence length: {max_length}")
print(f"Average sequence length: {average_length}")

X_text_train = pad_sequences(train_data['embeddings'].apply(lambda x: [embedding for embedding in x]), maxlen=max_length, dtype='float32')
X_numerical_train = train_data[numerical_columns].values
y_train = train_data['KTAS_expert'].values

X_text_val = pad_sequences(val_data['embeddings'].apply(lambda x: [embedding for embedding in x]), maxlen=max_length, dtype='float32')
X_numerical_val = val_data[numerical_columns].values
y_val = val_data['KTAS_expert'].values

# Apply SMOTE to balance the classes in the training data
smote = SMOTE()
X_text_train_resampled, y_train_resampled = smote.fit_resample(X_text_train.reshape((X_text_train.shape[0], -1)), y_train)
X_numerical_train_resampled, _ = smote.fit_resample(X_numerical_train, y_train)
X_text_train_resampled = X_text_train_resampled.reshape((-1, max_length, embedding_dim))

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
optimizer = Adam(learning_rate=0.0005)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Reduce learning rate on plateau
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Callback to save the best model based on validation accuracy
checkpoint = ModelCheckpoint('best_ktas_model.keras', monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)

# Train the model
history = model.fit([X_text_train_resampled, X_numerical_train_resampled], y_train_resampled,
                    epochs=50, batch_size=16, validation_data=([X_text_val, X_numerical_val], y_val),
                    shuffle=True, callbacks=[checkpoint, reduce_lr])

# Load the best model after training
model = tf.keras.models.load_model('best_ktas_model.keras')

# Plot accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'])

plt.show()

# Convert the class labels to string format
class_names = [str(label) for label in label_encoder.classes_]

# Evaluate the model on the test dataset
y_pred = model.predict([X_text_val, X_numerical_val])
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
print(classification_report(y_val, y_pred_classes, target_names=class_names))