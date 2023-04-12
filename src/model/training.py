from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import argparse
import glob
import os
import pandas as pd
import mlflow


def main(args):
    mlflow.autolog()
    df = read_data(args.training_data)
    X_train, X_test, y_train, y_test = split_data(df)
    train_model(args.reg_rate, X_train, y_train)


def read_data(path):
    if not os.path.exists(path):
        raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    try:
        df = pd.read_csv(path)
    except Exception as ex:
        mlflow.set_tag("exception", str(ex))
    return df


def split_data(data):
    X, y = data.loc[:, data.columns != 'target'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
    return X_train, X_test, y_train, y_test


def train_model(reg_rate, X_train, y_train):
    LogisticRegression(C=1/reg_rate, solver="liblinear").fit(X_train, y_train)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("\n\n")
    print("*" * 60)

    args = parse_args()
    main(args)

    print("*" * 60)
    print("\n\n")