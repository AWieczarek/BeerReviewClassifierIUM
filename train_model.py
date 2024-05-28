import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
import sys

train_data = pd.read_csv('beer_reviews_train.csv')
X_train = train_data[['review_aroma', 'review_appearance', 'review_palate', 'review_taste']]
y_train = train_data['review_overall']

tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train.values.ravel())
X_train_seq = tokenizer.texts_to_sequences(X_train.values.ravel())

X_train_pad = pad_sequences(X_train_seq, maxlen=100)

model = Sequential([
    Embedding(input_dim=10000, output_dim=16, input_length=100),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

epochs = int(sys.argv[1])
batch_size = int(sys.argv[2])
model.fit(X_train_pad, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1)

model.save('beer_review_sentiment_model.h5')
