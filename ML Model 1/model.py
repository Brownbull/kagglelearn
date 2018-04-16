import pandas as pd

# Input Data
melbourne_file_path = 'train.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data.columns)

# Predictions Location
y = melbourne_data.SalePrice

# Predictors Def
# melbourne_predictors = ['BedroomAbvGr', 'FullBath', 'LotArea', 'TotalBsmtSF',
#                         'YearBuilt']
melbourne_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Predictors Location
X = melbourne_data[melbourne_predictors]

# Model Fit imports
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Define Model
melbourne_model = DecisionTreeRegressor()

# Fit Model
melbourne_model.fit(train_X, train_y)
# melbourne_model.fit(X, y)
# print("Making predictions for the following 5 houses:")
# print(X.head())
# print("The predictions are")
# print(melbourne_model.predict(X.head()))

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))






