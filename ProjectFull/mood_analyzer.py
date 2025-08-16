import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords  # Remove 'emotion' from this import
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from collections import defaultdict
import emoji
class MoodAnalyzer:
    def __init__(self):
        # Download additional NLTK data
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')
        nltk.download('stopwords')
        nltk.download('vader_lexicon')
        
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            min_df=2,
            max_df=0.95,
            ngram_range=(1, 3),
            stop_words='english'
        )
        self.classifier = RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            random_state=42,
            class_weight='balanced'
        )
        self.label_encoder = LabelEncoder()
        self.stopwords = set(stopwords.words('english'))
        self.sia = SentimentIntensityAnalyzer()
        
        # Emotion keywords
        self.emotion_keywords = {
            'joy': ['happy', 'excited', 'delighted', 'pleased'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable'],
            'anger': ['angry', 'furious', 'irritated', 'annoyed'],
            'fear': ['scared', 'afraid', 'terrified', 'anxious'],
            'surprise': ['surprised', 'amazed', 'astonished', 'shocked'],
            'love': ['love', 'adore', 'cherish', 'passionate']
        }
        
    def detect_emotions(self, text):
        text_lower = text.lower()
        emotions = defaultdict(float)
        
        # Check for emotion keywords
        for emotion, keywords in self.emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    emotions[emotion] += 1
                    
        # Check for emojis
        for char in text:
            if char in emoji.EMOJI_DATA:
                emoji_name = emoji.EMOJI_DATA[char]['en'].lower()
                for emotion in self.emotion_keywords.keys():
                    if emotion in emoji_name:
                        emotions[emotion] += 1
        
        return dict(emotions)
        
    def preprocess_text(self, text):
        # Convert to lowercase
        text = text.lower()
        
        # Handle emojis
        text = emoji.demojize(text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # Tokenization
        tokens = word_tokenize(text)
        
        # Remove stopwords and short words
        tokens = [t for t in tokens if t not in self.stopwords and len(t) > 2]
        
        # Handle negations
        negations = ['not', 'no', 'never', "n't"]
        processed_tokens = []
        for i in range(len(tokens)):
            if i > 0 and tokens[i-1] in negations:
                processed_tokens.append(f'NOT_{tokens[i]}')
            else:
                processed_tokens.append(tokens[i])
        
        return ' '.join(processed_tokens)
        
    def train_model(self, training_data):
        # Preprocess the training texts
        processed_texts = training_data['text'].apply(self.preprocess_text)
        
        # Transform text data
        X = self.vectorizer.fit_transform(processed_texts)
        
        # Encode mood labels
        y = self.label_encoder.fit_transform(training_data['mood'])
        
        # Evaluate model performance
        cv_scores = cross_val_score(self.classifier, X, y, cv=5)
        print(f"Model Cross-validation Scores: {cv_scores.mean():.2f} (+/- {cv_scores.std() * 2:.2f})")
        
        # Train the classifier
        self.classifier.fit(X, y)
    
    def analyze_mood(self, text):
        # Preprocess input text
        processed_text = self.preprocess_text(text)
        
        # Basic sentiment analysis using TextBlob
        blob = TextBlob(text)
        sentiment_score = blob.sentiment.polarity
        subjectivity_score = blob.sentiment.subjectivity
        
        # VADER sentiment analysis
        vader_scores = self.sia.polarity_scores(text)
        
        # Emotion detection
        emotions = self.detect_emotions(text)
        
        # Advanced analysis using trained model
        text_vector = self.vectorizer.transform([processed_text])
        predicted_mood_encoded = self.classifier.predict(text_vector)[0]
        predicted_mood = self.label_encoder.inverse_transform([predicted_mood_encoded])[0]
        
        # Get probability distribution for all moods
        mood_probabilities = self.classifier.predict_proba(text_vector)[0]
        mood_confidence = dict(zip(
            self.label_encoder.classes_,
            [float(p) for p in mood_probabilities]
        ))
        
        # Enhanced mood mapping with more granular categories
        mood_mapping = {
            (-1.0, -0.7): "Very Negative",
            (-0.7, -0.4): "Negative",
            (-0.4, -0.1): "Slightly Negative",
            (-0.1, 0.1): "Neutral",
            (0.1, 0.4): "Slightly Positive",
            (0.4, 0.7): "Positive",
            (0.7, 1.0): "Very Positive"
        }
        
        for (lower, upper), mood in mood_mapping.items():
            if lower <= sentiment_score <= upper:
                basic_mood = mood
                break
                
        return {
            'text': text,
            'processed_text': processed_text,
            'sentiment_score': sentiment_score,
            'subjectivity_score': subjectivity_score,
            'vader_sentiment': vader_scores,
            'basic_mood': basic_mood,
            'predicted_mood': predicted_mood,
            'confidence': max(mood_probabilities),
            'mood_probabilities': mood_confidence,
            'detected_emotions': emotions,
            'intensity': abs(sentiment_score),
            'language_features': {
                'word_count': len(text.split()),
                'sentence_count': len(blob.sentences),
                'avg_word_length': sum(len(word) for word in text.split()) / len(text.split()) if text.split() else 0
            }
        }