# Normalize our particular diabetes dataset for easy computation and faster loss minimization.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from conf import *
import pickle

# Reading the diabetes dataset
df = pd.read_csv(DIABETES_DATASET)

# Saving the features and outcomes as float32 type
features = df.drop('Outcome', 1).values.astype(np.float32)
outcomes = df['Outcome'].values.astype(np.float32)

# Normalizing the features and writing them to csv file
features_normalized = MinMaxScaler().fit_transform(features)
features_normalized = pd.DataFrame(features_normalized, index=None)
features_normalized.to_csv(FEATURES_CSV, header=False, index=False)

# Saving the max and min value to use while prediction
features_max_data = features.max(0)
features_min_data = features.min(0)

# One hot encoding the outcome dataset
# Example:
# if our outcome is 0, one_hot(0) -> [1, 0]
# Reversal is done by using, np.argmax([1, 0]) -> 0
outcomes_one_hot_encoded =  pd.get_dummies(outcomes)
outcomes_one_hot_encoded.to_csv(OUTCOMES_CSV, header=False, index=False)

# Using pickle to store the max and min values
with open('data/min_max.pkl', 'w') as file:
	pickle.dump(features_max_data, file)
	pickle.dump(features_min_data, file)





