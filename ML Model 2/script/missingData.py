import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer


### SETUP - Start ###
# Input Data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

# Criterias
# # predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
# Predictors Location
# train_X = train[predictor_cols]
# Predictions Location
target = train.SalePrice
predictor_cols = train.drop(['SalePrice'], axis=1)

numeric_predictors = predictor_cols.select_dtypes(exclude=['object'])
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(numeric_predictors,
                                                    target,
                                                    train_size=0.7,
                                                    test_size=0.3,
                                                    random_state=0)


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)
### SETUP - End ###

### FILTERS -  Start ###

  # # HANDLING NULLS
  # # DROP NAN COLUMNS
  # data_without_missing_values = test.dropna(axis=1)
  # # Drop sames cols on terst and train datasets
  # cols_with_missing = [col for col in train.columns
  #                      if train[col].isnull().any()]
  # redued_original_data = train.drop(cols_with_missing, axis=1)
  # reduced_test_data = test.drop(cols_with_missing, axis=1)

  # IMPUTATION, REPLACING NAN FOR SOME VALUE
  # from sklearn.preprocessing import Imputer
  # my_imputer = Imputer()
  # data_with_imputed_values = my_imputer.fit_transform(train)

  # IMPUTATION B
  # make copy to avoid changing original data (when Imputing)
  # new_data = train.copy()

  # # make new columns indicating what will be imputed
  # cols_with_missing = (col for col in new_data.columns
  #                     if new_data[c].isnull().any())
  # for col in cols_with_missing:
  #     new_data[col + '_was_missing'] = new_data[col].isnull()

  # # Imputation
  # my_imputer = Imputer()
  # new_data = my_imputer.fit_transform(new_data)

# Get Model Score from Dropping Columns with Missing Values
cols_with_missing = [col for col in X_train.columns 
                                if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test  = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# Get Model Score from Imputation
my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# Get Score from Imputation with Extra Columns Showing What Was Imputed
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns
                     if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col +
                         '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col +
                        '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))

### FILTERS -  End ###

### MODEL
# my_model = RandomForestRegressor()
# my_model.fit(train_X, train_y)
# # Treat the test data in the same way as training data. In this case, pull same columns.
# test_X = test[predictor_cols]
# # Use the model to make predictions
# predicted_prices = my_model.predict(test_X)
# # We will look at the predicted prices to ensure we have something sensible.
# print(predicted_prices)
# print(train.isnull().sum())

# Submit@Kaggle
# my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# # you could use any filename. We choose submission here
# my_submission.to_csv('submission.csv', index=False)
