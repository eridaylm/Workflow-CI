import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="telco_preprocessing")
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=12)
args = parser.parse_args()

train = pd.read_csv(f"{args.data_dir}/train.csv")
test = pd.read_csv(f"{args.data_dir}/test.csv")
X_train, y_train = train.drop(columns="Churn"), train["Churn"]
X_test, y_test = test.drop(columns="Churn"), test["Churn"]

with mlflow.start_run():
    model = RandomForestClassifier(n_estimators=args.n_estimators,
                                   max_depth=args.max_depth, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
    mlflow.log_metrics({
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1_score": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
    })
    mlflow.sklearn.log_model(model, artifact_path="model")
    print("Model tersimpan, akurasi:", accuracy_score(y_test, pred))