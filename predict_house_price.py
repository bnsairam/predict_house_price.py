import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Sample review data (replace with real dataset or API data)
data = {
    'review': [
        'Amazing product, works perfectly and fast delivery!',
        'Terrible quality, broke after one use.',
        'Okay, nothing special but does the job.',
        'Love it, highly recommend to everyone!',
        'Disappointed, not worth the price.',
        'Decent product, good for the cost.'
    ] * 167,  # Repeated to create ~1000 reviews
    'sentiment': ['positive', 'negative', 'neutral', 'positive', 'negative', 'neutral'] * 167
}
df = pd.DataFrame(data)

# Preprocess text
stop_words = set(stopwords.words('english'))
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['clean_review'] = df['review'].apply(preprocess_text)

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(df['clean_review']).toarray()
y = df['sentiment']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize sentiment distribution
plt.figure(figsize=(10, 6))
sns.countplot(x=y_test, hue=y_pred, palette='viridis')
plt.title('Sentiment Distribution: Actual vs Predicted')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.legend(title='Predicted')
plt.tight_layout()
plt.savefig('sentiment_distribution.png')
plt.show()

# Feature importance visualization (top 10 TF-IDF words)
feature_importance = pd.Series(np.abs(model.coef_[0]), index=vectorizer.get_feature_names_out())
top_features = feature_importance.nlargest(10)
plt.figure(figsize=(8, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
plt.title('Top 10 Words by Importance in Sentiment Prediction')
plt.xlabel('Coefficient Magnitude')
plt.tight_layout()
plt.savefig('word_importance.png')
plt.show()

# Bonus: Example code to load data from a CSV or mock API (uncomment to use)
"""
import requests
def fetch_review_data(api_url):
    response = requests.get(api_url)
    if response.status_code == 200:
        data = response.json()
        df_api = pd.DataFrame(data)
        return df_api
    return None

# Example usage
# api_url = 'https://example.com/review-data-api'
# review_data = fetch_review_data(api_url)
# if review_data is not None:
#     review_data['clean_review'] = review_data['review'].apply(preprocess_text)
#     X_new = vectorizer.transform(review_data['clean_review']).toarray()
#     predicted_sentiment = model.predict(X_new)
#     print(f'Predicted sentiments for new reviews: {predicted_sentiment}')
"""
