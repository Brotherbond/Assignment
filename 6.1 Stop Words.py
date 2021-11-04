# Stop Words

# Importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Importing the data
dataset = """ Hello Mr. Watson, how are you doing today? 
             The weather is awesome. The garden is Green. 
             We should go out for a walk."""

# Create set of English Stop Words
stop_words = set(stopwords.words('english'))
print(stop_words)

# Tokenize words
word_tokenize = word_tokenize(dataset)

# Remove Stop Words from dataset
filtered_sentences = []

for w in word_tokenize:
    if w not in stop_words:
        filtered_sentences.append(w)

print(filtered_sentences)