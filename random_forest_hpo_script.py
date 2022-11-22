import argparse
import random
import time

import optuna
import pandas as pd
import pymysql
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import Evaluation

pymysql.install_as_MySQLdb()


def main(config):
    df = pd.read_csv('data/static_dynamic.csv')
    df = df.dropna()  # drop 96 records without pedestrian count
    column = ['SA1_CODE21']
    df = df.drop(column, axis=1)

    scale = StandardScaler()
    df_sc = scale.fit_transform(df)
    df_sc = pd.DataFrame(df_sc, columns=df.columns)

    y = df_sc['tripDensity']
    X = df_sc.drop(['tripDensity'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=24)

    model = RandomForestRegressor(n_estimators=config["n_estimators"],
                                  criterion="squared_error",
                                  min_samples_leaf=config["min_sample_leaf"],
                                  min_samples_split=config["min_sample_split"],
                                  max_depth=config["max_depth"],
                                  max_features=config["max_features"])
    print("model is training")
    model.fit(X_train, y_train)

    # Training results
    print("Train Results")
    y_train_pred = model.predict(X_train)
    train_eval = Evaluation(y_train, y_train_pred)
    train_eval.print()

    print("\nTest Results")
    # Test results
    y_test_pred = model.predict(X_test)
    test_eval = Evaluation(y_test, y_test_pred)
    test_eval.print()

    return test_eval.mse


def objective(trial):
    params = dict()
    params["n_estimators"] = trial.suggest_int("n_estimators", 20, 1000)
    params["min_sample_leaf"] = trial.suggest_int("min_sample_leaf", 1, 20)
    params["min_sample_split"] = trial.suggest_int("min_sample_split", 2, 40)
    params["max_features"] = trial.suggest_float("max_features", 1, 1.0, step=0.1)
    params["max_depth"] = trial.suggest_int("max_depth", 0, 40)
    if params["max_depth"] == 0:
        params["max_depth"] = None

    print(f"Initiating Run {trial.number} with params : {trial.params}")

    loss = main(params)
    return loss


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--optuna-db", type=str, help="Path to the Optuna Database file",
                        default="sqlite:///optuna.db")
    parser.add_argument("-n", "--optuna-study-name", type=str, help="Name of the optuna study",
                        default="hiruni_random_forest")
    args = parser.parse_args()

    # wait for some time to avoid overlapping run ids when running parallel
    wait_time = random.randint(0, 10) * 3
    print(f"Waiting for {wait_time} seconds before starting")
    time.sleep(wait_time)

    study = optuna.create_study(direction="minimize",
                                study_name=args.optuna_study_name,
                                storage=args.optuna_db,
                                load_if_exists=True,
                                )
    study.optimize(objective, n_trials=300)
