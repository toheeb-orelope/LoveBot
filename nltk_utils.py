import nltk  # Import the Natural Language Toolkit library
import numpy as np  # Import the NumPy library for numerical operations

nltk.download("punkt")  # Download the 'punkt' tokenizer models for sentence splitting
nltk.download("punkt_tab")  # Download the 'punkt_tab' resource (may not exist in NLTK)
from nltk.stem.porter import (
    PorterStemmer,
)  # Import the PorterStemmer for stemming words

stemmer = PorterStemmer()  # Create an instance of the PorterStemmer


def tokenize(sentence):
    return nltk.word_tokenize(sentence)  # Split a sentence into a list of words/tokens


def stem(word):
    return stemmer.stem(word.lower())  # Convert word to lowercase and return its stem


def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [
        stem(w) for w in tokenized_sentence
    ]  # Stem each word in the sentence
    bag = np.zeros(
        len(all_words), dtype=np.float32
    )  # Initialize a bag of words vector with zeros
    for idx, w in enumerate(all_words):
        if w in tokenized_sentence:  # If the word is present in the tokenized sentence
            bag[idx] = 1.0  # Set the corresponding position in the bag to 1
    return bag  # Return the bag of words vector
