# Twitter-Sentiment-Analysis-Analyzing-Tweets-for-Keyword-based-Sentiment-Classification

This class, called SentimentAnalyzer, is designed to analyze tweets by searching for specific keywords and determining the sentiment associated with those tweets.

It utilizes the Twitter API to fetch tweets containing the specified keywords. The train_model() method is responsible for training the sentiment analysis model. It reads negative and positive phrases from training files, combines them into a single dataset, and vectorizes the phrases using CountVectorizer. The model is then trained using a Support Vector Machine (SVM) algorithm.

The analyze_sentiment() method takes a tweet as input and performs sentiment analysis on it. It uses the trained model to vectorize the tweet and makes a probability prediction to determine the sentiment. The sentiment is returned as an emoji, indicating whether the tweet is positive or negative.

In summary, this code allows for sentiment analysis of tweets by searching for specific keywords and determining the sentiment associated with those tweets using a trained machine learning model.
