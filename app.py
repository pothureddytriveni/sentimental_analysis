# Import necessary libraries
from flask import Flask, request, render_template
import joblib

# Initialize Flask application
app = Flask(__name__)

# Load the trained models
sentiment_model = joblib.load('support_vector_machine_model.pkl')

# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for sentiment analysis
@app.route('/analyze_sentiment', methods=['POST'])
def analyze_sentiment():
    # Get input text from form
    text = request.form['text']

    # Perform sentiment analysis using the loaded model
    sentiment = sentiment_model.predict([text])[0]

    # Determine the sentiment result
    if sentiment == 1:
        result = 'Positive'
    else:
        result = 'Negative'

    # Render the result using a template named 'sentiment_result.html'
    return render_template('sentiment_result.html', result=result)


# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0",port=8080)
