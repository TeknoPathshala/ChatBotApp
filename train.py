# Import necessary libraries
import random
from tensorflow.keras.optimizers import SGD
from keras.layers import Dense, Dropout
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer

# Initialize WordNetLemmatizer from NLTK
lemmatizer = WordNetLemmatizer()

# Download NLTK data
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")

# Initialize lists and variables for data processing
words = []
classes = []
documents = []
ignore_words = ["?", "!"]

# Read intents from intents.json
data_file = open("intents.json").read()
intents = json.loads(data_file)

# Process intents and extract words and classes
for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        # Tokenize and lemmatize each word in the pattern
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent["tag"]))

        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# Lemmatize words and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Debugging: Print some statistics about the data
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

# Save words and classes to pickle files
pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# Initialize training data
training = []
output_empty = [0] * len(classes)

# Create training data
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# Shuffle training data and convert to NumPy array
random.shuffle(training)
training = np.array(training)

# Split training data into features and labels
train_x = list(training[:, 0])
train_y = list(training[:, 1])

# Create a neural network model
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))

# Compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

# Train the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model
model.save("chatbot_model.h5")
print("Model created and saved")
