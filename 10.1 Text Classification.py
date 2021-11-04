# Text Classification

# Import libraries
import nltk
import random
from nltk.corpus import movie_reviews

# Create a list of tuples
documents = []

for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((list(movie_reviews.words(fileid)), category))

# Shuffle the documents
random.shuffle(documents)
print(documents[0])

# Normalize the dataset
all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

# NLTK frequency distribution
all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))
print(all_words['love'])

# Limit the words
word_featuers = list(all_words.keys())[:3000]

# Find Features within the documents

def find_featuers(document):
    words = set(document)
    featuers = {}
    for w in word_featuers:
        featuers[w] = (w in words)
    return featuers

print((find_featuers(movie_reviews.words('pos/cv000_29590.txt'))))

featuresets = [(find_featuers(rev), category) for (rev, category) in documents]

# Split the dataset into Train and Test set
train_set = featuresets[:1900]
test_set = featuresets[1900:]

# Training the Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

# Test Accuracy
print('Accuracy:', 
      (nltk.classify.accuracy(classifier, test_set))*100)

classifier.show_most_informative_features(15)