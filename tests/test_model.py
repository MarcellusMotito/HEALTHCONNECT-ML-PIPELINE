import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def test_model_training():
    # Minimal dataset for testing
    X = pd.DataFrame({"feature1":[0,1,0,1], "feature2":[1,0,1,0]})
    y = [0,1,0,1]

    # Logistic Regression
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X,y)
    assert len(log_reg.predict(X)) == len(y)

    # Random Forest
    rf = RandomForestClassifier()
    rf.fit(X,y)
    assert len(rf.predict(X)) == len(y)

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X,y)
    assert len(xgb.predict(X)) == len(y)
