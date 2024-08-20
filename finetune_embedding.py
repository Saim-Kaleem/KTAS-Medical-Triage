import pandas as pd
import re
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from abbreviations import abbreviation_dict

# Load and preprocess the dataset
data = pd.read_csv('data/data_cleaned2.csv', on_bad_lines='skip')

# Visualize the vocabulary
def visualize_vocab(model, title):
    vocab = list(model.key_to_index.keys())[:100]  # Only plotting the first 100 words for readability
    vectors = model[vocab]
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(vectors)
    plt.figure(figsize=(12, 12))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], c='blue')
    for i, word in enumerate(vocab):
        plt.annotate(word, xy=(reduced_vectors[i, 0], reduced_vectors[i, 1]))
    plt.title(title)
    plt.show()

# Clean the text data
def clean_text(text):
    if isinstance(text, str):
        text = text.replace(',', '').replace('?', '')
        words = text.split()
        words = [abbreviation_dict.get(word, word) for word in words]
        text = ' '.join(words).lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
    return text

data['tokens'] = data['Chief_complain'].apply(clean_text).apply(lambda x: x.split())

# Load pre-trained embeddings
biowordvec_model = KeyedVectors.load_word2vec_format('embeddings/Bio_embedding_extrinsic', binary=True)
visualize_vocab(biowordvec_model, "Vocabulary Before Fine-Tuning")

# Create an initial Word2Vec model with the same vector size as the pre-trained embeddings
w2v_model = Word2Vec(vector_size=biowordvec_model.vector_size, window=5, min_count=1)
w2v_model.build_vocab(data['tokens'])

# Load the pre-trained BioWordVec embeddings into the Word2Vec model
w2v_model.build_vocab([list(biowordvec_model.key_to_index.keys())], update=True)
w2v_model.wv.vectors = biowordvec_model.vectors.copy()
w2v_model.wv.index_to_key = biowordvec_model.index_to_key.copy()
w2v_model.wv.key_to_index = biowordvec_model.key_to_index.copy()
w2v_model.wv.vectors_norm = biowordvec_model.vectors_norm.copy()

# Fine-tune the model on your data
w2v_model.train(data['tokens'], total_examples=w2v_model.corpus_count, epochs=10)
visualize_vocab(w2v_model.wv, "Vocabulary After Fine-Tuning")

# Save the fine-tuned embeddings to a new file
w2v_model.wv.save('embeddings/biowordvec_finetuned.kv')