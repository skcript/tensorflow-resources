# Normalize our particular dataset for easy computation and faster loss minimization.
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from conf import *
import pickle

# Reading the dataset
df = pd.read_csv(HOUSE_DATASET, sep=' ', header=None)
df.drop(df.columns[0],1, inplace=True)

# Saving the features and outcomes as float32 type
features = df.drop(df.columns[13], 1).values.astype(np.float32)

outcomes = df[13].values.astype(np.float32)
# Normalizing the features and writing them to csv file
features_normalized = MinMaxScaler().fit_transform(features)
features_normalized = pd.DataFrame(features_normalized, index=None)
features_normalized.to_csv(FEATURES_CSV, header=False, index=False)

# Writing them to csv file
outcomes = pd.DataFrame(outcomes, index=None)
outcomes.to_csv(OUTCOMES_CSV, header=False, index=False)

# Saving the max and min value to use while prediction
features_max_data = features.max(0)
features_min_data = features.min(0)

# Using pickle to store the max and min values
with open('data/min_max.pkl', 'w') as file:
	pickle.dump(features_max_data, file)
	pickle.dump(features_min_data, file)





