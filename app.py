from flask import Flask, render_template, request
from textblob import TextBlob

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if request.method == 'POST':
        text = request.form['text']
        analysis, emoji = analyze_sentiment(text)
        return render_template('index.html', text=text, analysis=analysis, sentiment_emoji={analysis: emoji})

def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_score = blob.sentiment.polarity

    if sentiment_score > 0:
        return 'Positive', '😊'
    elif sentiment_score < 0:
        return 'Negative', '😞'
    else:
        return 'Neutral', '😐'

if __name__ == '__main__':
    app.run(debug=True)
