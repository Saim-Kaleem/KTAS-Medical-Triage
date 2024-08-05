import pandas as pd
import re
from abbreviations import abbreviation_dict

# Load your data
data = pd.read_csv('data_cleaned.csv', on_bad_lines='skip')
data = data[['Chief_complain', 'KTAS_expert']]

# Clean the text data
def clean_text(text):
    if isinstance(text, str):  # Ensure text is a string
        # Remove commas and question marks
        text = text.replace(',', '').replace('?', '')

        # Replace abbreviations
        words = text.split() # Split text into words
        words = [abbreviation_dict.get(word, word) for word in words]
        text = ' '.join(words) # Join words back into a sentence
        
        # Convert to lowercase and remove any other unwanted characters
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
    return text

data['Chief_complain'] = data['Chief_complain'].apply(clean_text)
print(data.head())

# Save the processed data
data.to_csv('data_cleaned2.csv', index=False)