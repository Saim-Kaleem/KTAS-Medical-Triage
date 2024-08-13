import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Dense, Dropout, Concatenate, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from transformers import BertTokenizer, TFBertModel

# Load the training and validation datasets
train_data = pd.read_csv('data/ktas_train.csv', on_bad_lines='skip')
val_data = pd.read_csv('data/ktas_val.csv', on_bad_lines='skip')

# Initialize BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')

# Function to tokenize and create BERT embeddings
def bert_encode(texts, tokenizer, max_len=128):
    input_ids = []
    attention_masks = []
    for text in texts:
        bert_input = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='tf'
        )
        input_ids.append(bert_input['input_ids'])
        attention_masks.append(bert_input['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    return input_ids, attention_masks

# Preprocess and tokenize text data for training and validation
train_data['Chief_complain'] = train_data['Chief_complain'].apply(lambda x: x.lower())
val_data['Chief_complain'] = val_data['Chief_complain'].apply(lambda x: x.lower())

train_input_ids, train_attention_masks = bert_encode(train_data['Chief_complain'], tokenizer)
val_input_ids, val_attention_masks = bert_encode(val_data['Chief_complain'], tokenizer)

# Get BERT embeddings
train_bert_output = bert_model(train_input_ids, attention_mask=train_attention_masks).last_hidden_state
train_bert_output = tf.reduce_mean(train_bert_output, axis=1)

val_bert_output = bert_model(val_input_ids, attention_mask=val_attention_masks).last_hidden_state
val_bert_output = tf.reduce_mean(val_bert_output, axis=1)

# Convert KTAS_expert to numeric labels for training and validation data
label_encoder = LabelEncoder()
train_data['KTAS_expert'] = label_encoder.fit_transform(train_data['KTAS_expert'])
val_data['KTAS_expert'] = label_encoder.transform(val_data['KTAS_expert'])

# Normalize numerical data for training and validation
scaler = StandardScaler()
numerical_columns = ['Sex', 'Age', 'Arrival mode', 'Injury', 'Mental', 'Pain', 'BP', 'HR', 'RR', 'BT', 'Saturation']
train_data[numerical_columns] = scaler.fit_transform(train_data[numerical_columns])
val_data[numerical_columns] = scaler.transform(val_data[numerical_columns])

X_numerical_train = train_data[numerical_columns].values
X_numerical_val = val_data[numerical_columns].values
y_train = train_data['KTAS_expert'].values
y_val = val_data['KTAS_expert'].values

# Define the model architecture
numerical_input_shape = (len(numerical_columns),)
text_inputs = Input(shape=(train_bert_output.shape[-1],), name='text_input')
numerical_inputs = Input(shape=numerical_input_shape, name='numerical_input')

# Concatenate text and numerical inputs
combined_inputs = Concatenate()([text_inputs, numerical_inputs])
combined_inputs = BatchNormalization()(combined_inputs)

# Fully connected layers
fc_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(combined_inputs)
fc_layer = Dropout(0.4)(fc_layer)
fc_layer = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(fc_layer)
fc_layer = Dropout(0.4)(fc_layer)

# Output layer with softmax activation
output_layer = Dense(5, activation='softmax')(fc_layer)

# Build and compile the model
model = Model(inputs=[text_inputs, numerical_inputs], outputs=output_layer)
optimizer = Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Convert the TensorFlow tensor to numpy for model fitting
X_text_train = train_bert_output.numpy()
X_text_val = val_bert_output.numpy()

# Custom training loop
num_epochs = 50
batch_size = 16
initial_epochs = 8

# Calculate class weights
class_weights_dict = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_dict))

history = {
    'accuracy': [],
    'val_accuracy': [],
    'loss': [],
    'val_loss': []
}

# Initial training with class weights
history_initial = model.fit(
    [X_text_train, X_numerical_train], y_train,
    epochs=initial_epochs,
    batch_size=batch_size,
    validation_data=([X_text_val, X_numerical_val], y_val),
    shuffle=True,
    class_weight=class_weights_dict
)

# Append initial history
for key in history:
    history[key].extend(history_initial.history[key])

# Apply SMOTE to the training data
X_resampled, y_resampled = SMOTE().fit_resample(
    np.hstack((X_text_train, X_numerical_train)), y_train
)
X_text_resampled = X_resampled[:, :X_text_train.shape[1]].reshape(-1, X_text_train.shape[1])
X_numerical_resampled = X_resampled[:, X_text_train.shape[1]:]

# Continue training without class weights
history_smote = model.fit(
    [X_text_resampled, X_numerical_resampled], y_resampled,
    epochs=num_epochs - initial_epochs,
    batch_size=batch_size,
    validation_data=([X_text_val, X_numerical_val], y_val),
    shuffle=True
)

# Append SMOTE history
for key in history:
    history[key].extend(history_smote.history[key])

# Save the model
model.save('BERT_model.keras')

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

# Evaluate the model on the validation dataset to get precision, recall, and f1-score
y_pred = model.predict([X_text_val, X_numerical_val])
y_pred_classes = np.argmax(y_pred, axis=1)

# Print classification report
class_names = [str(label) for label in label_encoder.classes_]
print(classification_report(y_val, y_pred_classes, target_names=class_names))