from flask import Flask, request, render_template, jsonify
from mood_analyzer import MoodAnalyzer
import pandas as pd
app = Flask(__name__)
analyzer = MoodAnalyzer()
# Load and train the model with the larger dataset
sample_data = pd.read_csv('large_mood_dataset.csv')
analyzer.train_model(sample_data)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.json.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    result = analyzer.analyze_mood(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)