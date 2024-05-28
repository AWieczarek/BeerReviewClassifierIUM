import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

test_data = pd.read_csv('./beer_reviews_test.csv')
X_test = test_data[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']]
y_test = test_data['review_overall']

model = load_model('beer_review_sentiment_model.h5')

predictions = model.predict(X_test)
print(f'Predictions shape: {predictions.shape}')

if len(predictions.shape) > 1:
    predictions = predictions[:, 0]

results = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
results.to_csv('beer_review_sentiment_predictions.csv', index=False)
