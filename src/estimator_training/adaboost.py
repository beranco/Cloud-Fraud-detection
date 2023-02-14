#!/usr/bin/env python3

import os
import toml
import pickle
import warnings
import pandas as pd
import neptune.new as neptune
from toolkit import train, log_metrics
from sklearn.model_selection import RepeatedStratifiedKFold


# import estimators
from sklearn.ensemble import AdaBoostClassifier


# load featurization params
env = toml.load("env.toml")
warnings.filterwarnings("ignore")

# Import feature data
X_train = pd.read_csv("data/features/X_train.csv", sep=",")
X_test = pd.read_csv("data/features/X_test.csv", sep=",")
y_train = pd.read_csv("data/features/y_train.csv", sep=",")
y_test = pd.read_csv("data/features/y_test.csv", sep=",")

# Instantiate adaboost models
adaboost = AdaBoostClassifier()

# initialize neptune credentials
run = neptune.init_run(
    project=env["tracking"]["PROJECT_NAME"],
    api_token=env["tracking"]["NEPTUNE_API_TOKEN"],
    tags=["Adaboost Classifier"],
    capture_hardware_metrics=False,
    source_files=["model.py, featurize.py, train.py, env.toml"],
)

skfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)

# Train the model
train_adaboost = train(
    X_train=X_train,
    y_train=y_train,
    model=adaboost,
    skfold=skfold,
    logger=run,
    predict_proba=True,
)["model"]

# Log train metrics
log_metrics(
    model=train_adaboost,
    features=X_train,
    labels=y_train,
    filename="adaboost_train_conf_matrix",
    logger=run,
    log_type="train",
    predict_proba=True,
)
# Log adaboost metrics
log_metrics(
    model=train_adaboost,
    features=X_test,
    labels=y_test,
    filename="adaboost_eval_conf_matrix",
    logger=run,
    log_type="eval",
    predict_proba=True,
)

# Log all model params
run["model_params"] = adaboost.get_params()

# Export trained model
os.makedirs("models", exist_ok=True)
pickle.dump(train_adaboost, open("models/adaboost.pkl", "wb"))
run["models"].upload("models/adaboost.pkl")
run["scaled"] = env["featurize"]["SCALE"]

# Track training data
run["train_dataset/X_train"].track_files("data/features/X_train.csv")
run["train_dataset/y_train"].track_files("data/features/y_train.csv")
run["train_dataset/y_train"].track_files("notebooks/analysis.ipynb")

run.stop()
