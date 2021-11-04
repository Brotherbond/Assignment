# Named Entity Recognition

# Importing libraries
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk

# Importing the data
dataset = """Abraham Lincoln was an American statesman and lawyer 
              who served as the 16th President of the United States"""

# Tokenization and POS tagging
dataset_tag = pos_tag(word_tokenize(dataset))

# Apply NER
dataset_ner = ne_chunk(dataset_tag)
print(dataset_ner)

# Tree Diagram
dataset_ner.draw()
