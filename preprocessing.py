import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from gensim.models import KeyedVectors
from tensorflow.keras.preprocessing.sequence import pad_sequences
from abbreviations import abbreviation_dict

df = pd.read_csv('data/data_cleaned2.csv', on_bad_lines='skip')
df.drop(columns=['Diagnosis in ED'], axis=1, inplace=True)
print(df['KTAS_expert'].value_counts())

# Number of words/unique words
total_words =  ' '.join(df['Chief_complain'].values)
unique_words = set(total_words.lower().split())

print(f"Total words: {len(total_words.split())} | Unique words: {len(unique_words)}")

# Counting each word
word_count = pd.Series(total_words.split()).value_counts()
print(word_count.sort_values(ascending=False)) # Sorted from most frequent

# Get a list of our sentences lengths
sentence_len = []
for sentence in df["Chief_complain"]:
    sentence_len.append(len(sentence.split()))
print(sentence_len[:10])

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

df['Chief_complain'] = df['Chief_complain'].apply(clean_text)

# Get a list of our cleaned sentences lengths
sentence_len = []
for sentence in df["Chief_complain"]:
    sentence_len.append(len(sentence.split()))
print(sentence_len[:10])

# Max length and average length of the sequences
max_length = max(sentence_len)
average_length = np.mean(sentence_len)
print(f"Maximum sequence length: {max_length}")
print(f"Average sequence length: {average_length}")

# Zoom in distributions
plt.figure(figsize=(6,6))
plt.title("Sentences lengths distributions (zoom in)", fontsize=17)
plt.hist(sentence_len, bins=10, range=(0, 10))
plt.xlabel("Sentence lengths", fontsize=15)
plt.ylabel("Number of sentences", fontsize=15)
plt.show()

# Tokenize the text data
df['tokens'] = df['Chief_complain'].apply(lambda x: x.split())
print(df['tokens'].shape)
print(df['tokens'][0])

# Load the pre-trained word embeddings
biowordvec_model = KeyedVectors.load_word2vec_format('embeddings/bio_embedding_extrinsic', binary=True)
embedding_dim = biowordvec_model.vector_size

# Map tokens to embeddings
def get_embeddings(tokens, model, vector_size):
    embeddings = [model[word] if word in model else np.zeros(vector_size) for word in tokens]
    return np.array(embeddings)

df['embeddings'] = df['tokens'].apply(lambda x: get_embeddings(x, biowordvec_model, embedding_dim))
print(df['embeddings'].shape)
print(df['embeddings'][0])

X_text = pad_sequences(df['embeddings'], maxlen=max_length, padding='post', dtype='float32')
print(X_text[0])

# Padding sequences for PyTorch
def pad_sequence(sequences, maxlen, embedding_dim):
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

X_text2 = pad_sequence(df['embeddings'].tolist(), max_length, embedding_dim)
print(X_text2[0])