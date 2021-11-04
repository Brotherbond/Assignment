# Stemming

# Importing libraries
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

# Importing dataset
dataset = ['love', 'loving', 'lover', 'loved', 'lovingly']
new_dataset =  """ It feels very special when you are loving someone.
                  We care for our loved ones.
                  Specially when we love each other unconditionally """

# Apply stemming
ps = PorterStemmer()
for i in dataset:
    print(ps.stem(i))

words = word_tokenize(new_dataset)

for w in words:
    print(ps.stem(w))
