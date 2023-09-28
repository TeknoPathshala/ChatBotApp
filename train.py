# Import necessary libraries
import random
import tensorflow as tf
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Download NLTK data
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# Initialize NLTK's WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
with open('intents.json') as file:
    intents = json.load(file)

# Extract data from intents
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Save words and classes to files
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Create training data
training = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])]
