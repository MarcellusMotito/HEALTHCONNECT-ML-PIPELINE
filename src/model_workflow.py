from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def train_models(X_train, y_train, random_state=42):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(random_state=random_state)
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=random_state)

    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return log_reg, rf, xgb

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    return {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist()
    }
