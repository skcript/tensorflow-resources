import os 
filepath = os.path.abspath(__file__)
PWDPATH = os.path.dirname(filepath)

DATAPATH = os.path.join(PWDPATH , 'train.txt')
MODEL_FILE = 'trained_model/model.ckpt'
DATA_DIR = 'data'
DIABETES_DATASET = os.path.join(PWDPATH , DATA_DIR , 'dataset.data')
FEATURES_CSV = os.path.join(PWDPATH , DATA_DIR , 'features_normalized.csv')
OUTCOMES_CSV  = os.path.join(PWDPATH , DATA_DIR , 'outcome.csv')
PICKLE_FILE = os.path.join(PWDPATH, DATA_DIR, 'min_max.pkl')
SPLIT = 600
EPOCHS = 10