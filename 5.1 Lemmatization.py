# Lemmatization

# Importing libraries
import nltk
from nltk.stem import WordNetLemmatizer

# Apply Lemmatization
wnl = WordNetLemmatizer()

wnl.lemmatize('churches')
wnl.lemmatize('dogs')
wnl.lemmatize('feet')
wnl.lemmatize('better', pos='a')
