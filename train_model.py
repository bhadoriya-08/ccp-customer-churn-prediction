import os
import joblib
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from data_prep import load_and_clean

def train(data_path="./data/Telco-Customer-Churn.csv", model_path="./models/rf_churn.joblib"):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    df = load_and_clean(data_path)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    X = df.drop(columns=['Churn'])
    numeric_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object','category']).columns.tolist()
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_cols), ('cat', categorical_transformer, categorical_cols)])
    clf = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_score = clf.predict_proba(X_test)[:,1]
    print(classification_report(y_test, y_pred, digits=4))
    print("ROC AUC:", roc_auc_score(y_test, y_score))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    joblib.dump(clf, model_path)
    print("✅ Model saved to", model_path)
    meta = {"numeric_cols": numeric_cols, "categorical_cols": categorical_cols, "X_columns": X.columns.tolist()}
    with open(os.path.join(os.path.dirname(model_path),'model_metadata.json'), 'w') as f:
        json.dump(meta, f, indent=2)
    print("✅ Metadata saved.")

if __name__ == "__main__":
    train()
