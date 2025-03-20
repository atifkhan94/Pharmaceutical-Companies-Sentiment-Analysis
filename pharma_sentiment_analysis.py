import tweepy
import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Twitter API credentials
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'

# Initialize NLTK components
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class PharmaSentimentAnalyzer:
    def __init__(self):
        # Authenticate with Twitter
        auth = tweepy.OAuthHandler(API_KEY, API_SECRET_KEY)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        self.api = tweepy.API(auth, wait_on_rate_limit=True)
        
        self.companies = ['Bayer', 'Pfizer', 'Roche', 'Novartis']
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = CountVectorizer(max_features=5000)
        
    def clean_tweet(self, tweet):
        # Remove URLs
        tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
        # Remove user mentions
        tweet = re.sub(r'@\w+', '', tweet)
        # Remove hashtags
        tweet = re.sub(r'#\w+', '', tweet)
        # Remove numbers
        tweet = re.sub(r'\d+', '', tweet)
        # Convert to lowercase
        tweet = tweet.lower()
        # Tokenize
        tokens = word_tokenize(tweet)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        return ' '.join(tokens)
    
    def get_tweet_sentiment(self, tweet):
        analysis = TextBlob(tweet)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'
    
    def fetch_tweets(self, company, count=100):
        tweets = []
        try:
            fetched_tweets = self.api.search_tweets(q=company, lang='en', count=count)
            for tweet in fetched_tweets:
                parsed_tweet = {}
                parsed_tweet['text'] = self.clean_tweet(tweet.text)
                parsed_tweet['sentiment'] = self.get_tweet_sentiment(tweet.text)
                parsed_tweet['company'] = company
                if tweet.retweet_count > 0:
                    if parsed_tweet not in tweets:
                        tweets.append(parsed_tweet)
                else:
                    tweets.append(parsed_tweet)
            return tweets
        except tweepy.TweepError as e:
            print(f"Error: {str(e)}")
            return None
    
    def analyze_sentiments(self):
        all_tweets = []
        for company in self.companies:
            tweets = self.fetch_tweets(company)
            if tweets:
                all_tweets.extend(tweets)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_tweets)
        
        # Prepare features and labels
        X = self.vectorizer.fit_transform(df['text']).toarray()
        y = df['sentiment']
        
        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Naive Bayes classifier
        classifier = MultinomialNB()
        classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = classifier.predict(X_test)
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Visualize results
        self.plot_results(df)
    
    def plot_results(self, df):
        plt.figure(figsize=(12, 6))
        
        # Create sentiment distribution plot
        sentiment_counts = df.groupby(['company', 'sentiment']).size().unstack()
        sentiment_counts.plot(kind='bar', stacked=True)
        plt.title('Sentiment Distribution by Company')
        plt.xlabel('Company')
        plt.ylabel('Number of Tweets')
        plt.legend(title='Sentiment')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png')
        plt.close()

def main():
    analyzer = PharmaSentimentAnalyzer()
    analyzer.analyze_sentiments()

if __name__ == "__main__":
    main()