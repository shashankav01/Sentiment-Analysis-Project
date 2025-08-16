🎭 Mood Analyzer  

A machine learning–powered web application that analyzes user text and predicts the underlying **mood/emotion** using NLP techniques, sentiment analysis, and a trained classifier.  

🚀 Features  
- Text preprocessing with **NLTK** (tokenization, stopwords, negation handling).  
- **Sentiment analysis** using **TextBlob** and **VADER**.  
- Emotion keyword and emoji detection.  
- Machine learning model (**Random Forest Classifier** with TF-IDF features).  
- Flask-based API (`/analyze`) for mood detection.  
- Supports probability distribution of all mood classes for confidence scoring.  

 🛠️ Tech Stack  
- **Backend**: Flask (Python)  
- **ML/NLP**: scikit-learn, NLTK, TextBlob, Emoji  
- **Dataset**: Custom large mood dataset (`large_mood_dataset.csv`)  
- **Deployment Ready**: Gunicorn, Flask-CORS, dotenv  

📂 Project Structure  
```
├── app.py                 # Flask web server
├── mood_analyzer.py       # Core MoodAnalyzer class (ML + NLP logic)
├── large_mood_dataset.csv # Training dataset
├── requirements.txt       # Dependencies
├── project_report.tex     # Project report (LaTeX)
└── templates/
    └── index.html         # Frontend (if any UI added)
```

 ⚙️ Installation & Setup  

1. **Clone the repository**  
```bash
git clone https://github.com/yourusername/mood-analyzer.git
cd mood-analyzer
```

2. **Create and activate virtual environment**  
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Install dependencies**  
```bash
pip install -r requirements.txt
```

4. **Run the Flask app**  
```bash
python app.py
```

The server will start at `http://127.0.0.1:5000/`.  

📡 API Usage  

Endpoint: `/analyze`  
**Method:** `POST`  
**Request Body (JSON):**  
```json
{
  "text": "I am feeling really happy today!"
}
```

**Response:**  
```json
{
  "text": "I am feeling really happy today!",
  "processed_text": "feeling really happy today",
  "sentiment_score": 0.8,
  "subjectivity_score": 0.6,
  "vader_sentiment": {"pos": 0.9, "neu": 0.1, "neg": 0.0, "compound": 0.85},
  "basic_mood": "Positive",
  "predicted_mood": "joy",
  "confidence": 0.92,
  "mood_probabilities": {"joy": 0.92, "sadness": 0.03, "anger": 0.05},
  "detected_emotions": {"joy": 1},
  "intensity": 0.8,
  "language_features": {
    "word_count": 6,
    "sentence_count": 1,
    "avg_word_length": 4.1
  }
}
```

📊 Model Training  
- Preprocessing: lowercasing, stopword removal, emoji handling, negation handling.  
- Features: TF-IDF vectorizer (up to trigrams, 2000 features).  
- Model: Random Forest Classifier (balanced, depth 15, 300 trees).  
- Evaluation: 5-fold cross-validation printed on training.  

📖 Future Improvements  
- Add frontend UI for real-time mood detection.  
- Support for multi-lingual input.  
- Deployment to **Heroku / Render / AWS**.  
- Visualization dashboard for moods over time.  

👨‍💻 Author  
Developed by **[Shashank AV]**.  
