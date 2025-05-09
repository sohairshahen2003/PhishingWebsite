import os
import json
import numpy as np
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
from flask import Flask, render_template, request, url_for

app = Flask(__name__)

# Base path for models (adjust as needed for your system)
BASE_PATH = '/mnt/c/Users/dell/PycharmProjects/pro/test_results/'

# Load models with detailed logging
models = {}
try:
    for model_name in ['ann', 'cnn', 'rnn']:
        model_path = f"{BASE_PATH}{model_name}_model_all.h5"
        print(f"Checking for {model_name.upper()} model at: {model_path}")
        if os.path.exists(model_path):
            print(f"Found {model_name.upper()} model file. Attempting to load...")
            try:
                models[model_name] = keras.models.load_model(model_path)
                print(f"Successfully loaded {model_name.upper()} model")
            except Exception as e:
                print(f"Failed to load {model_name.upper()} model: {str(e)}")
        else:
            print(f"Error: {model_name.upper()} model file not found at {model_path}")
except Exception as e:
    print(f"Unexpected error while loading models: {e}")

# Load Tokenizer
char_index_path = f"{BASE_PATH}char_index.json"
if os.path.exists(char_index_path):
    print(f"Loading char_index from: {char_index_path}")
    with open(char_index_path, 'r') as f:
        char_index = json.load(f)
    tokenizer = Tokenizer(char_level=True, lower=True, oov_token='-n-')
    tokenizer.word_index = char_index
    sequence_length = 200
    print("Tokenizer loaded successfully")
else:
    print(f"Error: char_index.json not found at {char_index_path}")
    tokenizer, sequence_length = None, None

# Preprocess link for prediction
def preprocess_link(link):
    if not tokenizer or not sequence_length:
        raise ValueError("Tokenizer or sequence length not initialized")
    sequences = tokenizer.texts_to_sequences([link])
    padded_sequences = sequence.pad_sequences(sequences, maxlen=sequence_length)
    return padded_sequences

# Predict with a model
def predict_link(model, link):
    try:
        processed_link = preprocess_link(link)
        prediction = model.predict(processed_link, verbose=0)
        result = 'Phishing' if prediction[0][1] > 0.5 else 'Legitimate'
        confidence = prediction[0][1] if result == 'Phishing' else prediction[0][0]
        return result, confidence
    except Exception as e:
        return f"Error: {str(e)}", 0

# Home page
@app.route('/', methods=['GET', 'POST'])
def home():
    results = None
    error = None
    if request.method == 'POST':
        link = request.form.get('url', '').strip()
        if not link:
            error = "Please enter a valid URL!"
        else:
            results = {}
            for model_name in ['ann', 'cnn', 'rnn']:
                model = models.get(model_name)
                if model is None:
                    print(f"Model {model_name.upper()} is not available for prediction")
                    results[model_name] = {'result': 'Model not available', 'confidence': 0}
                else:
                    try:
                        print(f"Predicting with {model_name.upper()} model for URL: {link}")
                        result, confidence = predict_link(model, link)
                        results[model_name] = {'result': result, 'confidence': confidence}
                    except Exception as e:
                        print(f"Prediction error with {model_name.upper()}: {str(e)}")
                        results[model_name] = {'result': f"Error: {str(e)}", 'confidence': 0}
    return render_template('index.html', results=results, error=error)

# Scan route (for base.html default form)
@app.route('/scan', methods=['POST'])
def scan():
    results = None
    error = None
    link = request.form.get('url', '').strip()
    if not link:
        error = "Please enter a valid URL!"
    else:
        results = {}
        for model_name in ['ann', 'cnn', 'rnn']:
            model = models.get(model_name)
            if model is None:
                print(f"Model {model_name.upper()} is not available for prediction")
                results[model_name] = {'result': 'Model not available', 'confidence': 0}
            else:
                try:
                    print(f"Predicting with {model_name.upper()} model for URL: {link}")
                    result, confidence = predict_link(model, link)
                    results[model_name] = {'result': result, 'confidence': confidence}
                except Exception as e:
                    print(f"Prediction error with {model_name.upper()}: {str(e)}")
                    results[model_name] = {'result': f"Error: {str(e)}", 'confidence': 0}
    return render_template('index.html', results=results, error=error)

# About page
@app.route('/about')
def about():
    return render_template('about.html')

# ANN page
@app.route('/ann', methods=['GET', 'POST'])
def ann():
    result = None
    confidence = None
    error = None
    if request.method == 'POST':
        link = request.form.get('url', '').strip()
        if not link:
            error = "Please enter a valid URL!"
        else:
            model = models.get('ann')
            if model is None:
                print("ANN model is not available for prediction")
                error = "ANN model not available!"
            else:
                try:
                    print(f"Predicting with ANN model for URL: {link}")
                    result, confidence = predict_link(model, link)
                except Exception as e:
                    print(f"Prediction error with ANN: {str(e)}")
                    error = f"Prediction error: {str(e)}"
    return render_template('ann.html', result=result, confidence=confidence, error=error)

# CNN page
@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    result = None
    confidence = None
    error = None
    if request.method == 'POST':
        link = request.form.get('url', '').strip()
        if not link:
            error = "Please enter a valid URL!"
        else:
            model = models.get('cnn')
            if model is None:
                print("CNN model is not available for prediction")
                error = "CNN model not available!"
            else:
                try:
                    print(f"Predicting with CNN model for URL: {link}")
                    result, confidence = predict_link(model, link)
                except Exception as e:
                    print(f"Prediction error with CNN: {str(e)}")
                    error = f"Prediction error: {str(e)}"
    return render_template('cnn.html', result=result, confidence=confidence, error=error)

# RNN page
@app.route('/rnn', methods=['GET', 'POST'])
def rnn():
    result = None
    confidence = None
    error = None
    if request.method == 'POST':
        link = request.form.get('url', '').strip()
        if not link:
            error = "Please enter a valid URL!"
        else:
            model = models.get('rnn')
            if model is None:
                print("RNN model is not available for prediction")
                error = "RNN model not available!"
            else:
                try:
                    print(f"Predicting with RNN model for URL: {link}")
                    result, confidence = predict_link(model, link)
                except Exception as e:
                    print(f"Prediction error with RNN: {str(e)}")
                    error = f"Prediction error: {str(e)}"
    return render_template('rnn.html', result=result, confidence=confidence, error=error)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)