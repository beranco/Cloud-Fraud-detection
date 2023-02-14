#!/usr/bin/env python3

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime as dt
import neptune.new as neptune
import pandas as pd
import json
import toml
import os

os.makedirs(os.path.join("data", "features"), exist_ok=True)

# Import the prepared data
prepared_data = pd.read_csv("./data/prepared/prepared.csv")


# Based on previous analysis carried out the dataset class distribution is hightly imbalanced
normal_transactions = prepared_data[prepared_data["Class"] == 0]
fraud_transactions = prepared_data[prepared_data["Class"] == 1]

# Down sample the dataset
normal_transactions = normal_transactions.sample(n=508)
prepared_data = pd.concat([normal_transactions, fraud_transactions], axis=0)


# load featurization params
env = toml.load("env.toml")

# Split into features and labels
X = prepared_data.drop("Class", axis=1)
y = prepared_data["Class"]

# instantiate the encoder
X = pd.get_dummies(X)

# Perform feature selection
anova_filter = SelectKBest(f_classif, k=env["featurize"]["FEATURE_SIZE"])
anova_filter.fit_transform(X, y)
selected_features = list(anova_filter.get_feature_names_out())
X_feature_extracted = X[selected_features]


# Scale the data
if env["featurize"]["SCALE"]:
    scaler = StandardScaler()
    X_feature_extracted = pd.DataFrame(
        scaler.fit_transform(X_feature_extracted), columns=selected_features
    )
    print(200 * "***********")
    print("\t \t \t DATA SCALED \t \t")
    print(200 * "***********")

# split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_feature_extracted,
    y,
    test_size=env["featurize"]["SPLIT"],
    random_state=env["featurize"]["SEED"],
    stratify=y,
)


# Export splitted data
X_train.to_csv("data/features/X_train.csv", index=False)
X_test.to_csv("data/features/X_test.csv", index=False)
y_train.to_csv("data/features/y_train.csv", index=False)
y_test.to_csv("data/features/y_test.csv", index=False)

# log params
os.makedirs("data/logs", exist_ok=True)

data_artifacts = {
    "feature_size": env["featurize"]["FEATURE_SIZE"],
    "split_size": env["featurize"]["SPLIT"],
    "seed": env["featurize"]["SEED"],
    "scaled_or_normalized": env["featurize"]["SCALE"],
    "selected_features": list(selected_features),
    "before_feature_selection": X.shape,
    "after_feature_selection": X_feature_extracted.shape,
    "date": dt.strftime(dt.now(), "%Y-%m-%d:%H:%M"),
}

with open("data/logs/artifacts.json", "w") as file:
    file.write(json.dumps(data_artifacts, indent=4))
