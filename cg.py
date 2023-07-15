import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

class SentimentAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.accuracy = None

    def train_model(self):
        # Read the negative phrases file
        with open('training/negative_phrases.txt', 'r', encoding='utf-8') as file:
            sad_lines = file.readlines()

        # Read the positive phrases file
        with open('training/positive_phrases.txt', 'r', encoding='utf-8') as file:
            happy_lines = file.readlines()

        # Remove the newline character ("\n") at the end of each line
        sad_lines = [line.rstrip() for line in sad_lines]
        happy_lines = [line.rstrip() for line in happy_lines]

        # Create DataFrames for sad and happy phrases
        sad_data = pd.DataFrame({'phrase': sad_lines, 'label': 'sad'})
        happy_data = pd.DataFrame({'phrase': happy_lines, 'label': 'happy'})

        # Combine the DataFrames of sad and happy phrases
        data = pd.concat([sad_data, happy_data])

        # Create an instance of CountVectorizer to vectorize the phrases into numerical features
        self.vectorizer = CountVectorizer()

        # Vectorize the training phrases
        X_train_vectorized = self.vectorizer.fit_transform(data['phrase'])

        # Create a SVM (Support Vector Machine) model with probability support
        self.model = SVC(probability=True)

        # Train the model
        self.model.fit(X_train_vectorized, data['label'])

        # Calculate the model accuracy
        y_train_pred = self.model.predict(X_train_vectorized)
        self.accuracy = accuracy_score(data['label'], y_train_pred)

    def analyze_sentiment(self, phrase):
        # Vectorize the phrase
        phrase_vectorized = self.vectorizer.transform([phrase])

        # Make probability prediction with the trained model
        probabilities = self.model.predict_proba(phrase_vectorized)[0]

        positive_prob = probabilities[0]
        negative_prob = probabilities[1]

        # Get the final result (positive or negative)
        result = '🙂' if positive_prob >= negative_prob else '😥'

        return phrase, positive_prob, negative_prob, result
