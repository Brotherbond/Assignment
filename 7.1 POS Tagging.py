# Parts of Speech Tagging

# Importing libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

# Importing dataset
dataset = """Taj Mahal is one of the world’s most celebrated structures 
          in the world.
          It is a stunning symbol of Indian rich history"""

# Tokenize dataset
new_dataset = word_tokenize(dataset)
new_dataset

# Applying POS tagging
pos_tag(new_dataset)

# Tag set
nltk.help.upenn_tagset()