# Pharmaceutical Companies Sentiment Analysis

This project analyzes Twitter sentiment towards major pharmaceutical companies using Natural Language Processing (NLP) and Machine Learning techniques. It collects tweets about specific pharmaceutical companies, processes the text data, performs sentiment analysis, and visualizes the results.

## Features

- Real-time Twitter data collection for pharmaceutical companies
- Text preprocessing and cleaning
- Sentiment analysis using TextBlob
- Machine learning classification using Naive Bayes
- Visualization of sentiment distribution
- Handles retweets and duplicate content

## Companies Analyzed

- Bayer
- Pfizer
- Roche
- Novartis

## Requirements

All dependencies are listed in `requirements.txt`. Main requirements include:

- Python 3.7+
- tweepy
- pandas
- numpy
- textblob
- scikit-learn
- nltk
- matplotlib
- seaborn

## Installation

1. Clone the repository:
```bash
git clone [your-repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Twitter API Setup

1. Create a Twitter Developer account at https://developer.twitter.com
2. Create a new app in the Twitter Developer Portal
3. Generate API keys and tokens
4. Update the following variables in `pharma_sentiment_analysis.py`:
```python
API_KEY = 'your_api_key'
API_SECRET_KEY = 'your_api_secret_key'
ACCESS_TOKEN = 'your_access_token'
ACCESS_TOKEN_SECRET = 'your_access_token_secret'
```

## Usage

Run the main script:
```bash
python pharma_sentiment_analysis.py
```

The script will:
1. Fetch tweets about each pharmaceutical company
2. Clean and process the tweet text
3. Perform sentiment analysis
4. Train a Naive Bayes classifier
5. Generate a classification report
6. Create a visualization saved as 'sentiment_distribution.png'

## Output

- Console output: Classification report showing precision, recall, and F1-score for each sentiment category
- Visual output: A stacked bar chart showing sentiment distribution for each company saved as 'sentiment_distribution.png'

## Code Structure

- `pharma_sentiment_analysis.py`: Main script containing the PharmaSentimentAnalyzer class
- `requirements.txt`: List of Python package dependencies
- `sentiment_distribution.png`: Generated visualization of results

## Features in Detail

### Text Preprocessing
- URL removal
- Username removal
- Hashtag removal
- Number removal
- Lowercase conversion
- Tokenization
- Stop word removal
- Lemmatization

### Sentiment Analysis
- TextBlob for initial sentiment scoring
- Naive Bayes classifier for enhanced classification
- Handles three sentiment categories: positive, negative, neutral

### Visualization
- Stacked bar chart showing sentiment distribution
- Company-wise comparison
- Clear color coding for different sentiments

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. Please comply with Twitter's terms of service and API usage guidelines when using this tool.