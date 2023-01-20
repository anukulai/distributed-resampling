import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from sklearn.model_selection import StratifiedKFold
from smogn import smoter

from src.relevance.phi import Phi
from src.sampling.mixed_sampling.distributed_smogn import DistributedSMOGN
from src.sampling.over_sampling.distributed_ros import DistributedROS
from src.sampling.under_sampling.distributed_rus import DistributedRUS

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

#EXPERIMENTS - NEW DATASETS

DATA_DIR = "new_data"
DATA_RAW_DIR = f"{DATA_DIR}/raw"
DATA_PROCESSED_DIR = f"{DATA_DIR}/processed"

RESULT_DIR = "new_results"
RESULT_EXECUTION_TIME_DIR = f"{RESULT_DIR}"
RESULT_PREDICTIVE_PERFORMANCE_DIR = f"{RESULT_DIR}/predictive_performance"

DATASETS = {
    # "boston": "HousValue",
    "Abalone": "Rings",
    # "bank8FM": "rej",
    # "heat": "heat",
    "cpuSm": "usr",
    "energy": "Appliances",
    # "superconductivity": "critical_temp",
    # "creditScoring": "NumberOfDependents",
    # "census": "Unemployment",
    # "onion_prices": "modal_price",
    # "delays": "perdelay",
    # "gpu_performance": "Run4(ms)",
    "CountyHousing": "Total_Sale_Price",
    "HousePrices": "Prices",
    # "salary_compensation": "TotalCompensation",
}

EXPERIMENTS = {
    "ros": {
        "name": "ROS",
        "type": "dist",
        "sampler": DistributedROS
    },
    "rus": {
        "name": "RUS",
        "type": "dist",
        "sampler": DistributedRUS
    },
    "smogn": {
        "name": "SMOGN",
        "type": "seq",
        "sampler": smoter
    },
    "dist_smogn": {
        "name": "Distributed SMOGN",
        "type": "dist",
        "sampler": DistributedSMOGN,
        "k_partitions": [2, 4, 8]
    },
}

EXPERIMENTS_PERFORMANCE = {
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
        "file_postfix": "_dist_smogn_2"
    },
    "dist_smogn_4": {
        "name": "Distributed SMOGN (k_partitions = 4)",
        "file_postfix": "_dist_smogn_4"
    },
    "dist_smogn_8": {
        "name": "Distributed SMOGN (k_partitions = 8)",
        "file_postfix": "_dist_smogn_8"
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

spark = SparkSession.builder.master('local[4]').appName('Distributed Resampling').getOrCreate()

execution_times = {}
results = {}

# adding debugging print statement

# adding exception handling for all the sampling techniques

# Running for 10 iterations

for iteration in range(10):


    for dataset, label_col in DATASETS.items():

        print(dataset)
        DATA_PROCESSED_TRAIN_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/train"
        DATA_PROCESSED_TEST_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/test"

        print("Reading DF")
        df = pd.read_csv(f"{DATA_RAW_DIR}/{dataset}.csv")

        print("Reading DF in spark!")
        df = spark.createDataFrame(df)

        print("Calculating Phi value")
        relevance_col = "phi"
        df = Phi(input_col=label_col, output_col=relevance_col).transform(df)

        print("Splitting and Pre-processing")
        X = df.toPandas()
        y = X[label_col]
        k_fold_obj = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        k_fold_splits = k_fold_obj.split(X,y)

        for i, (train_indices, test_indices) in enumerate(k_fold_splits):
            print(f"Fold {i}:")

            X_train = X.iloc[train_indices]
            X_test = X.iloc[test_indices]


            # Remove index column
            X_train.drop(columns=[relevance_col], inplace=True)
            if "Unnamed: 0" in X_train.columns:
                X_train.drop(columns=["Unnamed: 0"], inplace=True)
            if "Unnamed: 0" in X_test.columns:
                X_test.drop(columns=["Unnamed: 0"], inplace=True)

            train = spark.createDataFrame(pd.DataFrame(X_train))
            print(f"{X_train.shape}, {X_test.shape}")

            phi = X_test.pop(relevance_col)

            print("Saving the CSV files")
            X_test.to_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_Fold_{i}.csv", index=False)
            phi.to_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_phi.csv", index=False)
            X_train.to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_Fold_{i}.csv", index=False)

            if iteration == 0:
                if i == 0:
                    execution_times[dataset] = {}
                execution_times[dataset]["RUS_Fold " + str(i)] = 0
                execution_times[dataset]["ROS_Fold " + str(i)] = 0
                execution_times[dataset]["SMOGN_Fold " + str(i)] = 0
                execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold " + str(i)] = 0
                execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold " + str(i)] = 0
                execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold " + str(i)] = 0

            try:
                print("Initializing Distributed RUS")
                start_time = time.time()
                train_rus = DistributedRUS(label_col=label_col, k_partitions=1).transform(train)
                end_time = time.time()
                execution_times[dataset]["RUS_Fold "+str(i)] = execution_times[dataset]["RUS_Fold "+str(i)] + round(end_time - start_time, 3)
                train_rus.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_rus_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Exception found in Distributed RUS: {e}")

            try:
                print("Initializing Distributed ROS")
                start_time = time.time()
                train_ros = DistributedROS(label_col=label_col, k_partitions=1).transform(train)
                end_time = time.time()
                execution_times[dataset]["ROS_Fold "+str(i)] = execution_times[dataset]["ROS_Fold "+str(i)] + round(end_time - start_time, 3)
                train_ros.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_ros_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Exception found in Distributed ROS: {e}")

            try:
                print("Initializing SMOGN")
                start_time = time.time()
                train_smogn = smoter(data=train.toPandas(), y=label_col)
                end_time = time.time()
                execution_times[dataset]["SMOGN_Fold "+str(i)] = execution_times[dataset]["SMOGN_Fold "+str(i)] + round(end_time - start_time, 3)
                train_smogn.to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_smogn_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Exception found in SMOGN: {e}")

            try:
                print("Initializing Distributed SMOGN with 2 partitions!")
                start_time = time.time()
                train_dist_smogn_2 = DistributedSMOGN(label_col=label_col, k_partitions=2).transform(train)
                end_time = time.time()
                execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold "+str(i)] = execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold "+str(i)] + round(end_time - start_time, 3)
                train_dist_smogn_2.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_2_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Found exception in DistSMOGN-2P: {e}")

            try:
                print("Initializing Distributed SMOGN with 4 partitions!")
                start_time = time.time()
                train_dist_smogn_4 = DistributedSMOGN(label_col=label_col, k_partitions=4).transform(train)
                end_time = time.time()
                execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold "+str(i)] = execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold "+str(i)] + round(end_time - start_time, 3)
                train_dist_smogn_4.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_4_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Found exception in DistSMOGN-4P: {e}")

            try:
                print("Initializing Distributed SMOGN with 8 partitions!")
                start_time = time.time()
                train_dist_smogn_8 = DistributedSMOGN(label_col=label_col, k_partitions=8).transform(train)
                end_time = time.time()
                execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold "+str(i)] = execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold "+str(i)] + round(end_time - start_time, 3)
                train_dist_smogn_8.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_8_Fold_{i}.csv", index=False)
            except Exception as e:
                print(f"Found exception in Dist-SMOGN-8P: {e}")


            # Experiments to observe Performance on different regressors
            # for dataset, label_col in DATASETS.items():
            DATA_PROCESSED_TRAIN_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/train"
            DATA_PROCESSED_TEST_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/test"

            for regressor, regressor_config in REGRESSORS.items():

                print(dataset + " " + regressor + "###############################################")
                for experiment, experiment_config in EXPERIMENTS_PERFORMANCE.items():
                    print("------- Running EXPT loop again ------")
                    train = pd.read_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}{experiment_config['file_postfix']}_Fold_{i}.csv")
                    y_train = train.pop(label_col)
                    x_train = pd.get_dummies(train)

                    test = pd.read_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_Fold_{i}.csv")
                    y_test = test.pop(label_col)
                    x_test = pd.get_dummies(test)

                    # Remove index column
                    if "Unnamed: 0" in x_train.columns:
                        x_train.drop(columns=["Unnamed: 0"], inplace=True)
                    if "Unnamed: 0" in x_test.columns:
                        x_test.drop(columns=["Unnamed: 0"], inplace=True)

                    # x_train, x_test = x_train.align(x_test, join='outer', axis=1)  # outer join --RATHER ONE HOT ENCODER BEFORE FITTING
                    print("reading datasets done")
                    # Only fitting the MinMaxScalar on train and later transforming both train and test.
                    scaler = MinMaxScaler().fit(x_train)


                    x_train = scaler.transform(x_train)
                    x_test = scaler.transform(x_test)

                    y_phi = pd.read_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_phi.csv")

                    mae_list = []
                    rmse_list = []

                    results[experiment_config["name"]+"_Fold_"+str(i)] = {}

                    for model in regressor_config["variants"]:
                        model.fit(x_train, y_train)

                        y_pred = model.predict(x_test)

                        mae_list.append(mean_absolute_error(y_true=y_test, y_pred=y_pred, sample_weight=y_phi))
                        rmse_list.append(
                            mean_squared_error(y_true=y_test, y_pred=y_pred, sample_weight=y_phi, squared=False))

                    print("All Regressors done - "+dataset + " " + regressor)
                    results[experiment_config["name"]+"_Fold_"+str(i)]["mae"] = round(np.mean(mae_list), 3)
                    results[experiment_config["name"]+"_Fold_"+str(i)]["rmse"] = round(np.mean(rmse_list), 3)


    for folds in range(5):
        execution_times[dataset]["RUS_Fold " + str(i)] = (execution_times[dataset]["RUS_Fold "+str(i)])/10
        execution_times[dataset]["ROS_Fold " + str(i)] = (execution_times[dataset]["ROS_Fold "+str(i)])/10
        execution_times[dataset]["SMOGN_Fold " + str(i)] = (execution_times[dataset]["SMOGN_Fold "+str(i)])/10
        execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold " + str(i)] = (execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold "+str(i)])/10
        execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold " + str(i)] = (execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold "+str(i)])/10
        execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold " + str(i)] = (execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold "+str(i)])/10

pd.DataFrame(data=execution_times).to_csv(f"{RESULT_EXECUTION_TIME_DIR}/execution_time.csv", index=True)
pd.DataFrame(data=results).transpose().to_csv(f"{RESULT_PREDICTIVE_PERFORMANCE_DIR}/{dataset}/{regressor}.csv",index=True)