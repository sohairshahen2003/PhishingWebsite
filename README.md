PhishGuard is a web application designed to detect phishing URLs using advanced AI models, including Artificial Neural Networks (ANN), Convolutional Neural Networks (CNN), and Recurrent Neural Networks (RNN). Our goal is to provide a user-friendly tool to help individuals and organizations identify malicious URLs and stay safe online.
using 
# PhishGuard - Phishing URL Detection System

Welcome to **PhishGuard**, an AI-powered web application designed to detect phishing URLs using machine learning models. This project leverages Flask for the web framework and implements three different neural network architectures: Artificial Neural Network (ANN), Convolutional Neural Network (CNN), and Recurrent Neural Network (RNN) to analyze and classify URLs as either "Phishing" or "Legitimate".

## Features
- Scan URLs using ANN, CNN, or RNN models.
- Compare results from all three models on a single page.
- User-friendly interface built with Bootstrap.
- Real-time prediction with confidence scores.

## Prerequisites
Before running the project, ensure you have the following installed:
- Python 3.x
- TensorFlow
- Flask
- Pyngrok (for hosting on Colab or local tunneling)
- MLflow (optional, for tracking experiments)

You can install the required packages using:
```bash
pip install tensorflow flask pyngrok mlflow
