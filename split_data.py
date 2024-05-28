import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('data/beer_reviews.csv')

train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

train_data.to_csv('beer_reviews_train.csv', index=False)
test_data.to_csv('beer_reviews_test.csv', index=False)
