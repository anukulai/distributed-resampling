import argparse

import numpy as np
import logging
import pandas as pd
import warnings
warnings.filterwarnings(action="ignore")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

logging.basicConfig(level=20)
LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(level=20)

DATA_DIR = "new_data"   # change the data directory
DATA_RAW_DIR = f"{DATA_DIR}/raw"
DATA_PROCESSED_DIR = f"{DATA_DIR}/processed_faiss_kmeans"

RESULT_DIR = "results_faiss_kmeans"
RESULT_EXECUTION_TIME_DIR = f"{RESULT_DIR}"
RESULT_PREDICTIVE_PERFORMANCE_DIR = f"{RESULT_DIR}/predictive_performance"

RANDOM_STATES = [42, 99, 65, 100, 1, 7, 23, 67, 11, 97]
NUM_ITERATIONS = 10
NUM_FOLDS = 5

DATASETS = {
    "boston": "HousValue",
    "Abalone": "Rings",
    "bank8FM": "rej",
    "heat": "heat",
    "cpuSm": "usr",
    "energy": "Appliances",
    "superconductivity": "critical_temp",
    "creditScoring": "NumberOfDependents",
    "census": "Unemployment",
    "onion_prices": "modal_price",
    "gpu_performance": "Run4(ms)",
    "CountyHousing": "Total_Sale_Price",
    "HousePrices": "Prices"
}

EXPERIMENTS = {
    "base": {
        "name": "No Sampling",
        "file_postfix": ""
    },
    "rus": {
        "name": "RUS",
        "file_postfix": "_rus"
    },
    "ros": {
        "name": "ROS",
        "file_postfix": "_ros"
    },
    "smogn": {
        "name": "SMOGN",
        "file_postfix": "_smogn"
    },
    "dist_smogn_2": {
        "name": "Distributed SMOGN (k_partitions = 2)",
        "file_postfix": "_distsmogn2"
    },
    "dist_smogn_4": {
        "name": "Distributed SMOGN (k_partitions = 4)",
        "file_postfix": "_distsmogn4"
    },
    "dist_smogn_8": {
        "name": "Distributed SMOGN (k_partitions = 8)",
        "file_postfix": "_distsmogn8"
    },
    "dist_smogn_16": {
        "name": "Distributed SMOGN (k_partitions = 16)",
        "file_postfix": "_distsmogn16"
    }
}

REGRESSORS = {
    "lr": {
        "name": "Linear Regression (LR)",
        "variants": [
            LinearRegression()
        ]
    },
    "svm": {
        "name": "Support Vector Machine (SVM)",
        "variants": [
            SVR(C=10, gamma=0.01),
            SVR(C=10, gamma=0.001),
            SVR(C=150, gamma=0.01),
            SVR(C=150, gamma=0.001),
            SVR(C=300, gamma=0.01),
            SVR(C=300, gamma=0.001)
        ]
    },
    "rf": {
        "name": "Random Forest (RF)",
        "variants": [
            RandomForestRegressor(min_samples_leaf=1, min_samples_split=2),
            RandomForestRegressor(min_samples_leaf=1, min_samples_split=5),
            RandomForestRegressor(min_samples_leaf=2, min_samples_split=2),
            RandomForestRegressor(min_samples_leaf=2, min_samples_split=5),
            RandomForestRegressor(min_samples_leaf=4, min_samples_split=2),
            RandomForestRegressor(min_samples_leaf=4, min_samples_split=5)
        ]
    },
    "nn": {
        "name": "Neural Network (NN)",
        "variants": [
            MLPRegressor(hidden_layer_sizes=1, max_iter=500),
            MLPRegressor(hidden_layer_sizes=1, max_iter=1000),
            MLPRegressor(hidden_layer_sizes=5, max_iter=500),
            MLPRegressor(hidden_layer_sizes=5, max_iter=1000),
            MLPRegressor(hidden_layer_sizes=10, max_iter=500),
            MLPRegressor(hidden_layer_sizes=10, max_iter=1000)
        ]
    }
}

columns_to_remove = {
    "census": ["Unnamed: 0", "County", "State", "CensusTract"],
    "onion_prices": ["Unnamed: 0", "commodity", "arrival_date"],
    'CountyHousing': ["Unnamed: 0", "Real_Estate_Id", "Total_Sale_Date", "Month_Year_of_Sale", "Physical_Zip"],
    'gpu_performance': ["Run1(ms)", "Run2(ms)", "Run3(ms)"],
    'HousePrices': ["Unnamed: 0"],
    'Abalone': [],
    'boston': [],
    'bank8FM': [],
    'heat': [],
    'cpuSm': [],
    'energy': [],
    'superconductivity': [],
    'creditScoring': []
}

def run_folds(dataset_name, label_col, iteration):
    LOGGER.info(f"Running folds for {dataset_name} for {iteration} iteration!")
    for fold in range(NUM_FOLDS):
        DATA_PROCESSED_TRAIN_DIR = f"{DATA_PROCESSED_DIR}/{dataset_name}/train"
        DATA_PROCESSED_TEST_DIR = f"{DATA_PROCESSED_DIR}/{dataset_name}/test"

        for regressor, regressor_config in REGRESSORS.items():
            results = {}
            if regressor != regressor_name:
                continue

            for experiment, experiment_config in EXPERIMENTS.items():
                LOGGER.info(f"Iteration: {iteration} | Fold: {fold} | Regressor: {regressor} | Exp: {experiment}")
                train = pd.read_csv(
                    f"{DATA_PROCESSED_TRAIN_DIR}/{dataset_name}{experiment_config['file_postfix']}_iter_{iteration}_fold_{fold}.csv"
                )
                train.drop(columns=columns_to_remove[dataset_name], inplace=True)

                test = pd.read_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset_name}_iter_{iteration}_fold_{fold}.csv")
                test.drop(columns=columns_to_remove[dataset_name], inplace=True)

                y_train = train.pop(label_col)
                x_train = pd.get_dummies(train)

                y_test = test.pop(label_col)
                x_test = pd.get_dummies(test)

                x_train, x_test = x_train.align(x_test, join='inner', axis=1)

                scaler = MinMaxScaler().fit(x_train)

                x_train = scaler.transform(x_train)
                x_test = scaler.transform(x_test)

                y_phi = pd.read_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset_name}_phi_iter_{iteration}_fold_{fold}.csv")

                mae_list = []
                rmse_list = []

                results[experiment_config["name"] + "_iter_"+str(iteration) + "_fold_" + str(fold)] = {}

                for model in regressor_config["variants"]:
                    model.fit(x_train, y_train)

                    y_pred = model.predict(x_test)

                    mae_list.append(mean_absolute_error(y_true=y_test, y_pred=y_pred, sample_weight=y_phi))
                    rmse_list.append(
                        mean_squared_error(y_true=y_test, y_pred=y_pred, sample_weight=y_phi, squared=False))

                results[experiment_config["name"] + "_iter_"+str(iteration) + "_fold_" + str(fold)]["mae"] = round(np.mean(mae_list), 3)
                results[experiment_config["name"] + "_iter_"+str(iteration) + "_fold_" + str(fold)]["rmse"] = round(np.mean(rmse_list), 3)

            pd.DataFrame(data=results).transpose().to_csv(
                f"{RESULT_PREDICTIVE_PERFORMANCE_DIR}/{dataset_name}/{regressor}_iter_{iteration}_fold_{fold}.csv", index=True
            )

def run_iterations(dataset_name):

    LOGGER.info(f"Running {NUM_ITERATIONS} iterations for {dataset_name}")

    label_col = DATASETS.get(dataset_name)

    for iteration in range(NUM_ITERATIONS):
        run_folds(dataset_name, label_col, iteration)



def orchestrate_resampling(dataset_name):
    run_iterations(dataset_name)
    return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, required=True)
    arg_parser.add_argument("--regressor", type=str, required=True)

    arguments = arg_parser.parse_args()
    dataset_name = arguments.dataset
    regressor_name = arguments.regressor

    orchestrate_resampling(dataset_name)
