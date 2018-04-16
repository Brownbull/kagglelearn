import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

### SETUP - Start ###
# Input Data
melbourne_file_path = 'train.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# print(melbourne_data.columns)

# Criterias
# melbourne_predictors = ['BedroomAbvGr', 'FullBath', 'LotArea', 'TotalBsmtSF',
#                         'YearBuilt']
melbourne_predictors = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF',
                        'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

# Predictors Location
X = melbourne_data[melbourne_predictors]
# Predictions Location
y = melbourne_data.SalePrice

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
### SETUP - End ###

### MODEL
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))