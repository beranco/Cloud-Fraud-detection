#!/usr/bin/env python3

import os
import toml
import pickle
import warnings
import pandas as pd
import neptune.new as neptune
from sklearn.metrics import roc_auc_score
from toolkit import train, log_metrics
from sklearn.model_selection import RepeatedStratifiedKFold


# import estimators
from sklearn.ensemble import VotingClassifier


# load featurization params
env = toml.load("env.toml")
warnings.filterwarnings("ignore")

# Import feature data
X_train = pd.read_csv("data/features/X_train.csv", sep=",")
X_test = pd.read_csv("data/features/X_test.csv", sep=",")
y_train = pd.read_csv("data/features/y_train.csv", sep=",")
y_test = pd.read_csv("data/features/y_test.csv", sep=",")

# initialize neptune credentials
run = neptune.init_run(
    project=env["tracking"]["PROJECT_NAME"],
    api_token=env["tracking"]["NEPTUNE_API_TOKEN"],
    tags=["Final Estimator"],
    capture_hardware_metrics=False,
    source_files=["model.py, featurize.py, train.py, env.toml"],
)
# load models
train_adaboost = pickle.load(open("models/adaboost.pkl", "rb"))
train_rf_model = pickle.load(open("models/random_forest.pkl", "rb"))
train_knn = pickle.load(open("models/knn.pkl", "rb"))
train_log_model = pickle.load(open("models/logistic_reg.pkl", "rb"))

# compute the threshold from all four models
predictions = []
for model in [train_adaboost, train_rf_model, train_knn, train_log_model]:
    predictions.append(pd.Series(model.predict_proba(X_test)[:, 1]))

avg_predictions = pd.concat(predictions, axis=1).mean(axis=1)
avg_predictions.to_csv("data/prepared/average_predictions_per_model.csv")
run["train/avg_roc_auc_score"].log(roc_auc_score(y_test, avg_predictions))
run["model_predictions"].track_files("data/prepared/average_predictions_per_model.csv")

models = [
    ("adaboost", train_adaboost),
    ("rf_model", train_rf_model),
    ("knn", train_knn),
    ("lgr", train_log_model),
]

ensemble = VotingClassifier(models, voting="hard")

skfold = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)

train_ensemble = train(
    X_train=X_train,
    y_train=y_train,
    model=ensemble,
    skfold=skfold,
    logger=run,
    predict_proba=False,
)["model"]

# Log train metrics
log_metrics(
    model=train_ensemble,
    features=X_train,
    labels=y_train,
    filename="final_model_train_conf_matrix",
    logger=run,
    log_type="train",
    predict_proba=False,
)

log_metrics(
    model=train_ensemble,
    features=X_test,
    labels=y_test,
    filename="final_model_eval_conf_matrix",
    logger=run,
    log_type="eval",
    predict_proba=False,
)

# Log all model params
run["model_params"] = ensemble.get_params()

# Export trained model
os.makedirs("models", exist_ok=True)
pickle.dump(ensemble, open("models/final.pkl", "wb"))
run["models"].upload("models/final.pkl")
run["scaled"] = env["featurize"]["SCALE"]

# Track training data
run["train_dataset/X_train"].track_files("data/features/X_train.csv")
run["train_dataset/y_train"].track_files("data/features/y_train.csv")
run["train_dataset/y_train"].track_files("notebooks/analysis.ipynb")

run.stop()
