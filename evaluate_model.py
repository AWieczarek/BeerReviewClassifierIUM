import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

test_data = pd.read_csv('beer_reviews_test.csv')
X_test = test_data[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']]
y_test = test_data['review_overall']

model = load_model('beer_review_sentiment_model.h5')

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_test.values.ravel())
X_test_seq = tokenizer.texts_to_sequences(X_test.values.ravel())

X_test_pad = pad_sequences(X_test_seq, maxlen=100)

predictions = model.predict(X_test)
print(f'Predictions shape: {predictions.shape}')

if len(predictions.shape) > 1:
    predictions = predictions[:, 0]

results = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
results.to_csv('beer_review_sentiment_predictions.csv', index=False)
