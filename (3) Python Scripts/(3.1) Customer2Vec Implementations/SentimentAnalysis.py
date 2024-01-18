# TextBlob

#!pip install TextBlob
#!pip install vaderSentiment

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob

# VADER
def vader_sentiment_scores(text):
    vader_sentiment = SentimentIntensityAnalyzer()
    score = vader_sentiment.polarity_scores(text)
    return score['compound']

# TextBlob
def textblob_sentiment_scores(text):
    textblob_sentiment = TextBlob(text)
    score = textblob_sentiment.sentiment.polarity
    return score


def flair_sentiment_scores(text, sia):

    from flair.models import TextClassifier
    from flair.data import Sentence
    sentence = Sentence(text)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        return score.score
    elif "NEGATIVE" in str(score):
        return score.score * -1
    else:
        return score.score


def sentiment_ensemble(text,sia):

    from flair.models import TextClassifier
    from SentimentAnalysis import textblob_sentiment_scores
    from flair.data import Sentence
    sentence = Sentence(text)
    sia.predict(sentence)
    score = sentence.labels[0]
    if "POSITIVE" in str(score):
        flair = score.score
    elif "NEGATIVE" in str(score):
        flair = score.score * -1
    else:
        flair = score.score

    textblob = textblob_sentiment_scores(text)

    if textblob < 0.5 and textblob > -0.5:
        score = textblob

    if flair > abs(0.9):
        score = flair

    return score
