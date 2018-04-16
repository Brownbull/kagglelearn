import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

### SETUP - Start ###
# Input Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Criterias
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
# Predictors Location
train_X = train[predictor_cols]
# Predictions Location
train_y = train.SalePrice

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
### SETUP - End ###

### MODEL
my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)