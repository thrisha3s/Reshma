source code:
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Sample dataset (fake news data)
data = {
    'text': [
        "The Earth is flat.",
        "NASA's Mars rover successfully lands on Mars.",
        "A man was eaten by a giant shark off the coast of Australia.",
        "Local elections were held in New York City last week.",
        "A new health breakthrough claims to cure all forms of cancer.",
        "Scientists discover a new species in the Amazon rainforest.",
        "The Moon landing was staged and never happened.",
        "The stock market crashed due to unexpected news from the Federal Reserve.",
        "A person can live without food for 30 days but cannot survive without water for more than 3 days.",
        "The sky is green during the day and blue at night."
    ],
    'label': ['fake', 'real', 'fake', 'real', 'fake', 'real', 'fake', 'real', 'real', 'fake']
}

# Convert the dataset to a DataFrame
df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(X_train_tfidf, y_train)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)
score = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\n=== Model Performance ===")
print(f"Accuracy: {score * 100:.2f}%")
print("\nConfusion Matrix:")
print("              Predicted Fake  Predicted Real")
print(f"Actually Fake     {conf_matrix[0][0]}              {conf_matrix[0][1]}")
print(f"Actually Real     {conf_matrix[1][0]}              {conf_matrix[1][1]}")

print("\n=== Training Data Statistics ===")
print(f"Total training samples: {len(X_train)}")
print(f"Total test samples: {len(X_test)}")
print(f"Features used: {X_train_tfidf.shape[1]}")

def predict_news(statement):
    statement_tfidf = vectorizer.transform([statement])
    prediction = model.predict(statement_tfidf)
    # Get feature importance
    feature_importance = model.coef_[0]
    important_words = vectorizer.get_feature_names_out()[abs(feature_importance).argsort()[-5:]]
    return prediction[0], important_words

print("\n=== Test Predictions ===")
test_statements = [
    "Scientists discover a new species in the Amazon rainforest.",
    "Chocolate cures all diseases instantly.",
    "New study shows benefits of regular exercise.",
]

for statement in test_statements:
    prediction, important_words = predict_news(statement)
    print(f"\nStatement: '{statement}'")
    print(f"Prediction: {prediction}")
    print(f"Key words considered: {', '.join(important_words)}")
output:
=== Model Performance ===
Accuracy: 0.00%

Confusion Matrix:
              Predicted Fake  Predicted Real
Actually Fake     0              0
Actually Real     2              0

=== Training Data Statistics ===
Total training samples: 8
Total test samples: 2
Features used: 42

=== Test Predictions ===

Statement: 'Scientists discover a new species in the Amazon rainforest.'
Prediction: real
Key words considered: staged, landing, moon, flat, earth

Statement: 'Chocolate cures all diseases instantly.'
Prediction: fake
Key words considered: staged, landing, moon, flat, earth

Statement: 'New study shows benefits of regular exercise.'
Prediction: real
Key words considered: staged, landing, moon, flat, earth
