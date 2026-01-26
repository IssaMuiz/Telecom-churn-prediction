from pathlib import Path
import pandas as pd
from src.data_exploratory.clean_data import clean_data
from src.data_exploratory.split_data import split_data, split_features_target


# PATH
RAW_DATA_PATH = Path("data/raw/Telco_customer_churn.xlsx")
PROCESS_DATA_DIR = Path("data/processed/")
PROCESS_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load raw data
df = pd.read_excel(RAW_DATA_PATH, engine='openpyxl')

# Clean data
df = clean_data(df)

# Train, validation and test split
train_df, val_df, test_df = split_data(df)

# Split features and target for training set
X_train, y_train = split_features_target(train_df)
# Split features and target for validation set
X_val, y_val = split_features_target(val_df)
# Split features and target for test set
X_test, y_test = split_features_target(test_df)


X_train.to_csv(PROCESS_DATA_DIR / "X_train.csv", index=False)
y_train.to_csv(PROCESS_DATA_DIR / "y_train.csv", index=False)
X_val.to_csv(PROCESS_DATA_DIR / "X_val.csv", index=False)
y_val.to_csv(PROCESS_DATA_DIR / "y_val.csv", index=False)
X_test.to_csv(PROCESS_DATA_DIR / "X_test.csv", index=False)
y_test.to_csv(PROCESS_DATA_DIR / "y_test.csv", index=False)

print("Data cleaned and splitted successfully")
