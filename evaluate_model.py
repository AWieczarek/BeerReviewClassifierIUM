import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
from math import sqrt

test_data = pd.read_csv('beer_reviews_test.csv')
X_test = test_data[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']]
y_test = test_data['review_overall']

model = load_model('beer_review_sentiment_model.keras')

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_test)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_test_pad = pad_sequences(X_test_seq, maxlen=100)

predictions = model.predict(X_test)

if len(predictions.shape) > 1:
    predictions = predictions[:, 0]

results = pd.DataFrame({'Predictions': predictions, 'Actual': y_test})
results.to_csv('beer_review_sentiment_predictions.csv', index=False)

y_pred = results['Predictions']
y_test = results['Actual']
y_test_binary = (y_test >= 3).astype(int)

accuracy = accuracy_score(y_test_binary, y_pred.round())
precision, recall, f1, _ = precision_recall_fscore_support(y_test_binary, y_pred.round(), average='micro')
rmse = sqrt(mean_squared_error(y_test, y_pred))

print(f'Accuracy: {accuracy}')
print(f'Micro-avg Precision: {precision}')
print(f'Micro-avg Recall: {recall}')
print(f'F1 Score: {f1}')
print(f'RMSE: {rmse}')
