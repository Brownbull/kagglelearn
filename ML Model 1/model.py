import pandas as pd

melbourne_file_path = 'train.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.describe())