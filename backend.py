from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import numpy as np

def load_split_df(dataset: str):
    if dataset == "Mean imputation":
        df = pd.read_csv('data/mean_df')
        results = joblib.load('models/results_Mean')
        scaler = joblib.load('models/scaler_mean.pkl')
        X_mean = df.drop("Outcome", axis=1)
        y_mean = df['Outcome']

        _, X_mean_test, _, y_mean_test = train_test_split(
        X_mean, y_mean,
        test_size= 0.2,
        random_state=42,
        stratify=y_mean
        )

        return df, X_mean_test, y_mean_test, results, scaler
    else:
        df = pd.read_csv('data/mice_df')
        results = joblib.load('models/results_Mice')
        scaler = joblib.load('models/scaler_mice.pkl')
        X_mice = df.drop("Outcome", axis=1)
        y_mice = df['Outcome']

        _, X_mice_test, _, y_mice_test = train_test_split(
        X_mice, y_mice,
        test_size= 0.2,
        random_state=42,
        stratify=y_mice
        )

        return df, X_mice_test, y_mice_test, results, scaler
    
def plot_shap(model, X_test, model_name):
    if model_name == "RF" or model_name == "XGB":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        if isinstance(shap_values, list):
            shap_vals = shap_values[1]          # liste de 2 arrays → prendre classe 1
        elif len(np.array(shap_values).shape) == 3:
            shap_vals = shap_values[:, :, 1]    # shape (n, features, classes)
        else:
            shap_vals = shap_values

    else: # for knn and svm and xgb
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test, 50))
        shap_values = explainer.shap_values(X_test)
        shap_vals   = shap_values[1] if isinstance(shap_values, list) else shap_values

    print(f"shap_vals final shape : {np.array(shap_vals).shape}")

    fig, ax = plt.subplots()
    shap.summary_plot(shap_vals, X_test, show=False)
    return fig

    




