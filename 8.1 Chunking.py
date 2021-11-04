# Chunking

# Importing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import RegexpParser

# Importing the data
dataset = """Taj Mahal is one of the world’s most celebrated structures 
          in the world.
          It is a stunning symbol of Indian rich history"""

# Tokenize dataset
new_dataset = word_tokenize(dataset)

# Applying POS tagging
postagging = pos_tag(new_dataset)
postagging

# Define sequence of chunk
sequence_chunk = """ 
chunk:
    {<NNPS>+}
    {<NNP>+}
    {<NN>+} """

# Create object with Regular Expression
chunk = RegexpParser(sequence_chunk)

# Apply chunking
chunk_result = chunk.parse(postagging)
print(chunk_result)













