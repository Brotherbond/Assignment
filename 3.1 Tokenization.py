# Tokenization

# Importing libraries
import nltk
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

# Importing dataset
dataset = "Hello Everyone. Welcome to this course. We are studying NLP."

# Tokenizing sentences
print(sent_tokenize(text=dataset, language='english'))
for i in sent_tokenize(text=dataset, language='english'):
    print(i)

# Tokenizing words
print(word_tokenize(text=dataset, language='english'))
for i in word_tokenize(text=dataset, language='english'):
    print(i)
