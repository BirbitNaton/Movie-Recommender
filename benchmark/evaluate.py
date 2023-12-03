import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error


def get_metrics(y_true, y_pred, tag:str):
    y_true, y_pred = y_true*5, y_pred*5

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    mae_round = mean_absolute_error(y_true.round(), y_pred.round())
    mse_round = mean_squared_error(y_true.round(), y_pred.round())
    rmse_round = np.sqrt(mse_round)

    index = [(tag, "continous"),
             (tag, "discrete")]
    index = pd.MultiIndex.from_tuples(index, names=["model", "prediction_type"])

    results = pd.DataFrame({
        "MAE": [mae, mae_round],
        "MSE": [mse, mse_round],
        "RMSE": [rmse, rmse_round]
        },
        index=index)
    
    return results


def evluate():
    models_path = Path("models")
    interim_path = interim_path = Path("data/interim")

    model1 = joblib.load(models_path / "model_2080.pkl")
    model2 = joblib.load(models_path / "model_disjoint.pkl")

    X_test, y_test = pd.read_parquet(interim_path / "test_x1.parquet"), pd.read_parquet(interim_path / "test_y1.parquet")
    X_test2, y_test2 = pd.read_parquet(interim_path / "test_x2.parquet"), pd.read_parquet(interim_path / "test_y2.parquet")
    y_test = y_test.rating
    y_test2 = y_test2.rating

    predictions_2080 = model1.predict(X_test)
    predictions_disjoint = model2.predict(X_test2)

    results_2080 = get_metrics(y_test, predictions_2080, "2080")
    results_disjoint = get_metrics(y_test2, predictions_disjoint, "disjoint")
    benchmark = pd.concat([results_2080, results_disjoint])

    return benchmark


if __name__ == "__main__":
    benchmark = evluate()
    benchmark.to_csv("benchmark/data/benchmark.csv")
    print(benchmark)
