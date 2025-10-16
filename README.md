# Sentiment Analysis of Product Reviews

## Overview
This project predicts the sentiment (positive, negative, neutral) of product reviews using natural language processing (NLP) and machine learning. It employs a Logistic Regression model from scikit-learn with TF-IDF vectorization to classify sentiments based on review text. The project includes visualizations using seaborn to show sentiment distribution and word importance. A bonus section provides code to load data from a CSV or mock API endpoint.

## Technologies
- Python 3.8+
- scikit-learn
- pandas
- seaborn
- matplotlib
- nltk
- requests (for API integration, optional)

## Features
- Predicts review sentiment using a Logistic Regression model.
- Visualizes actual vs. predicted sentiment distribution with a count plot.
- Displays top words influencing sentiment based on model coefficients.
- Supports loading real-world review data from a CSV or API.
