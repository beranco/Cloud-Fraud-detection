from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    plot_roc_curve,
)
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def train(X_train, y_train, model, skfold, logger, predict_proba=True):

    fold_no = 1
    recall_scores = []
    f1_scores = []
    acc_scores = []
    roc_auc_scores = []

    for train_index, val_index in skfold.split(X_train, y_train):

        train_X, val_X = X_train.iloc[train_index], X_train.iloc[val_index]
        train_y, val_y = y_train.iloc[train_index], y_train.iloc[val_index]

        model.fit(train_X, train_y)
        if predict_proba:
            y_pred = model.predict_proba(val_X)
            roc_auc = roc_auc_score(val_y, y_pred[:, 1])
            roc_auc_scores.append(roc_auc)
            logger["train/roc_auc_score"].log(roc_auc)
            y_pred = np.argmax(y_pred, axis=1)
        else:
            y_pred = model.predict(val_X)

        re_score = recall_score(val_y, y_pred)
        f1 = f1_score(val_y, y_pred)
        acc_score = accuracy_score(val_y, y_pred)

        recall_scores.append(re_score)
        acc_scores.append(acc_score)
        f1_scores.append(f1)

        # Log metrics (series of values)
        logger["train/recall_score"].log(re_score)
        logger["train/f1_score"].log(f1)
        logger["train/accuracy"].log(acc_score)

        print(
            f"Fold {fold_no}| \
                Recall: {round(recall_scores[fold_no - 1], 2)} |-| \
                F1: {round(f1_scores[fold_no - 1], 2)} \
                ACC: {round(acc_scores[fold_no-1], 2)} ",
            f"ROC: {round(roc_auc_scores[fold_no-1], 2)}" if predict_proba else "",
        )

        fold_no += 1

    re_score, f1, acc_scores, roc_auc_scores = (
        np.mean(recall_scores),
        np.mean(f1_scores),
        np.mean(acc_scores),
        np.mean(roc_auc_scores),
    )

    plt.savefig("resources/plots/train_conf_matrix.png")
    logger["train/conf_matrix"].upload("resources/plots/train_conf_matrix.png")
    plt.figure()

    return {
        "model": model,
        "recall": re_score,
        "f1_score": f1,
        "accuracy": acc_scores,
        "roc": roc_auc_scores,
    }


def log_metrics(
    model, features, labels, filename, logger, log_type="eval", predict_proba=True
):

    if predict_proba:
        y_pred = model.predict_proba(features)
        roc_auc = roc_auc_score(labels, y_pred[:, 1])
        logger[f"{log_type}/roc_auc"].log(roc_auc)

        y_pred = np.argmax(y_pred, axis=1)

        plot_roc_curve(model, features, labels)
        plt.savefig(f"resources/plots/roc/{filename}")
        logger[f"{log_type}/roc_curv"].upload(f"resources/plots/roc/{filename}.png")
        plt.figure()

    else:

        y_pred = model.predict(features)

    re_score = recall_score(labels, y_pred)
    f1 = f1_score(labels, y_pred)
    acc_score = accuracy_score(labels, y_pred)

    # Log metrics (series of values)
    logger[f"{log_type}/recall_score"].log(re_score)
    logger[f"{log_type}/f1_score"].log(f1)
    logger[f"{log_type}/accuracy"].log(acc_score)

    conf_matrix = confusion_matrix(labels, y_pred)
    sns.heatmap(
        conf_matrix,
        annot=True,
        cmap="YlGnBu",
        fmt=".2f",
    )

    # You can upload image from the disc
    plt.savefig(f"resources/plots/conf_mat/{filename}")
    logger[f"{log_type}/conf_matrix"].upload(f"resources/plots/conf_mat/{filename}.png")
    plt.figure()
