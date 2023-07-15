from flask import Flask, request, jsonify, render_template
from cg import *
from cg_keys import *

app = Flask(__name__)

# Create an instance of SentimentAnalyzer
analyzer = SentimentAnalyzer()

# Load and train the model upon Flask application startup
@app.before_request
def load_and_train_model():
    analyzer.train_model()
    print('Model trained.')

# Route to render index.html
@app.route('/')
def index():
    return render_template('index.html')

# Route to classify tweets based on keywords
@app.route('/classify', methods=['POST'])
def classify_tweets():
    keywords = request.json['keywords']
    tweets = api.search_tweets(q=f'lang:en {keywords}', count=1)
    results = []

    for tweet in tweets:
        analyzed_post, positive_prob, negative_prob, result = analyzer.analyze_sentiment(tweet.text)
        positive_prob *= 100
        negative_prob *= 100

        result = {
            'tweet_content': analyzed_post,
            'positive_probability': f'{positive_prob:.2f}%',
            'negative_probability': f'{negative_prob:.2f}%',
            'result': result,
            'model_accuracy': f'{analyzer.accuracy:.2f}'
        }
        results.append(result)

    return jsonify(results)

# Route to render the /staff page
@app.route('/staff')
def staff():
    return render_template('staff.html')

# Route to add positive and negative phrases to training files
@app.route('/add_phrases', methods=['POST'])
def add_phrases():
    positive_phrase = request.json['positivePhrase']
    negative_phrase = request.json['negativePhrase']

    with open('training/positive_phrases.txt', 'a', encoding='utf-8') as file:
        file.write(positive_phrase + '\n')

    with open('training/negative_phrases.txt', 'a', encoding='utf-8') as file:
        file.write(negative_phrase + '\n')

    response = {
        'success': True,
        'message': 'Phrases added successfully!'
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run()
