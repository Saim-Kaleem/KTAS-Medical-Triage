import pandas as pd
import re
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, Dot, Softmax, Reshape, Multiply, Add, Concatenate, BatchNormalization, Bidirectional, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
from abbreviations import abbreviation_dict
from sklearn.utils import class_weight
from imblearn.over_sampling import SMOTE
from sklearn.utils import shuffle
from itertools import cycle

# Function to visualize attention scores
def plot_attention(text, attention_scores, max_len):
    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_subplot(111)
    if len(attention_scores.shape) == 1:
        attention_scores = np.expand_dims(attention_scores, axis=0)
    ax.matshow(attention_scores, cmap='viridis')
    ax.set_xticks(np.arange(len(text)))
    ax.set_yticks([0])
    ax.set_xticklabels(text, rotation=90)
    ax.set_yticklabels(['Attention'])
    plt.show()

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

# Train a Word2Vec model
# w2v_model = Word2Vec(sentences=data['tokens'], vector_size=50, window=3, min_count=1, workers=4)

# Load pretrained embeddings
biowordvec_model = KeyedVectors.load_word2vec_format('embeddings/bio_embedding_extrinsic', binary=True)

# Create an embedding matrix
# embedding_matrix = w2v_model.wv
# embedding_dim = embedding_matrix.vector_size
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

X_text = pad_sequences(data['embeddings'].apply(lambda x: [embedding for embedding in x]), maxlen=max_length, dtype='float32')
X_numerical = data[numerical_columns].values
y = data['KTAS_expert'].values

print(f"X_text shape: {X_text.shape}")
print(f"X_numerical shape: {X_numerical.shape}")
print(f"y shape: {y.shape}")

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_dict = dict(enumerate(class_weights))

# Define the model architecture
text_input_shape = (max_length, embedding_dim)
numerical_input_shape = (len(numerical_columns),)

# Stratified k-fold validation
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
all_histories = []
best_val_accuracies = []

# For ROC curves
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

for train_index, val_index in kfold.split(X_text, y):
    print(f"Training fold {fold_no}...")

    # Split the data
    X_text_train, X_text_val = X_text[train_index], X_text[val_index]
    X_numerical_train, X_numerical_val = X_numerical[train_index], X_numerical[val_index]
    y_train, y_val = y[train_index], y[val_index]

    # Create the model
    text_inputs = Input(shape=text_input_shape, name='text_input')
    numerical_inputs = Input(shape=numerical_input_shape, name='numerical_input')

    # CNN-based n-gram encoder for text inputs
    conv_layer = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(text_inputs)
    conv_layer = MaxPooling1D(pool_size=2)(conv_layer)
    conv_layer = Conv1D(filters=128, kernel_size=3, padding='same', activation='relu', kernel_initializer='he_normal')(conv_layer)
    conv_layer = MaxPooling1D(pool_size=2)(conv_layer)

    # Attention mechanism

    # Bidirectional LSTM layer
    lstm_layer = Bidirectional(LSTM(units=128, return_sequences=False, dropout=0.2, recurrent_dropout=0.2))(conv_layer)

    # Concatenate text and numerical inputs
    combined_inputs = Concatenate()([lstm_layer, numerical_inputs]) 

    # Apply batch normalization to the combined output
    combined_inputs = BatchNormalization()(combined_inputs)

    # Fully connected layers
    fc_layer = Dense(128, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(combined_inputs)
    fc_layer = Dropout(0.3)(fc_layer)
    fc_layer = Dense(64, activation='relu', kernel_regularizer=l2(0.01), kernel_initializer='he_normal')(fc_layer)
    fc_layer = Dropout(0.2)(fc_layer)

    # Output layer with softmax activation
    output_layer = Dense(5, activation='softmax')(fc_layer)

    # Build the model
    model = Model(inputs=[text_inputs, numerical_inputs], outputs=output_layer)
    if fold_no==1: model.summary()

    # Set an optimizer and compile the model
    optimizer = Adam(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Reduce learning rate on plateau
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

    # Save the best model based on validation accuracy
    checkpoint = ModelCheckpoint(
        f'ktas_model_fold{fold_no}.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    # Apply SMOTE
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(np.hstack((X_text_train.reshape(X_text_train.shape[0], -1), X_numerical_train)), y_train)
    X_text_resampled = X_resampled[:, :X_text_train.shape[1] * X_text_train.shape[2]].reshape(-1, X_text_train.shape[1], X_text_train.shape[2])
    X_numerical_resampled = X_resampled[:, X_text_train.shape[1] * X_text_train.shape[2]:]

    # Training with oversampled data
    history_initial = model.fit(
        [X_text_resampled, X_numerical_resampled], y_resampled,
        epochs=50, batch_size=16, 
        validation_data=([X_text_val, X_numerical_val], y_val), 
        shuffle=True, callbacks=[reduce_lr, checkpoint]
    )

    # Load the best model
    model = load_model(f'ktas_model_fold{fold_no}.keras')

    # Plot the training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history_initial.history['accuracy'], label='train')
    plt.plot(history_initial.history['val_accuracy'], label='validation')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history_initial.history['loss'], label='train')
    plt.plot(history_initial.history['val_loss'], label='validation')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Calculate classification report
    y_pred = model.predict([X_text_val, X_numerical_val])
    y_pred_classes = np.argmax(y_pred, axis=1)
    print(f"Classification report for fold {fold_no}:")
    print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_.astype(str)))

    # Calculate and print confusion matrix
    conf_matrix = confusion_matrix(y_val, y_pred_classes)
    ConfusionMatrixDisplay(conf_matrix, display_labels=label_encoder.classes_).plot(cmap='Blues')
    plt.title(f'Confusion Matrix for fold {fold_no}')
    plt.show()

    # Calculate specificity for each class
    tn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - np.diag(conf_matrix)
    fn = conf_matrix.sum(axis=1) - np.diag(conf_matrix)
    tp = np.diag(conf_matrix)
    specificity = tn / (tn + fp)
    print(f'Specificity for fold {fold_no}: {specificity}')

    # Compute ROC curve and AUC for this fold
    y_prob = model.predict([X_text_val, X_numerical_val])
    fpr, tpr, _ = roc_curve(y_val, y_prob[:, 1], pos_label=1)
    tprs.append(np.interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)

    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold_no} (AUC = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

    fold_no += 1

# Plot the ROC curve for each fold along with the mean ROC curve
plt.figure(figsize=(10, 8))

# Plot the ROC curve for each fold
for i, (tpr, auc_val) in enumerate(zip(tprs, aucs)):
    plt.plot(mean_fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {i + 1} (AUC = {auc_val:.2f})')

# Plot the mean ROC curve
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)

plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'± 1 std. dev.')

# Plot the luck line
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()