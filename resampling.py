import argparse
import time

import pandas as pd
from pyspark.sql import SparkSession
from sklearn.model_selection import KFold
from smogn import smoter

from src.relevance.phi import Phi
from src.sampling.mixed_sampling.distributed_smogn import DistributedSMOGN
from src.sampling.over_sampling.distributed_ros import DistributedROS
from src.sampling.under_sampling.distributed_rus import DistributedRUS


DATA_DIR = "new_data"
DATA_RAW_DIR = f"{DATA_DIR}/raw"
DATA_PROCESSED_DIR = f"{DATA_DIR}/processed"

RESULT_DIR = "new_results"
RESULT_EXECUTION_TIME_DIR = f"{RESULT_DIR}"
RESULT_PREDICTIVE_PERFORMANCE_DIR = f"{RESULT_DIR}/predictive_performance"

DATASET_TO_LABEL_MAP = {
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
    "delays": "perdelay",
    "gpu_performance": "Run4(ms)",
    "CountyHousing": "Total_Sale_Price",
    "HousePrices": "Prices",
    "salary_compensation": "TotalCompensation",
}

RANDOM_STATES = [42, 99, 65, 100, 1, 7, 23, 67, 11, 97]
NUM_ITERATIONS = 10
NUM_FOLDS = 5

EXECUTION_TIME = {}

#
# EXPERIMENTS = {
#     # "ros": {
#     #     "name": "ROS",
#     #     "type": "dist",
#     #     "sampler": DistributedROS
#     # },
#     # "rus": {
#     #     "name": "RUS",
#     #     "type": "dist",
#     #     "sampler": DistributedRUS
#     # },
#     # "smogn": {
#     #     "name": "SMOGN",
#     #     "type": "seq",
#     #     "sampler": smoter
#     # },
#     "dist_smogn": {
#         "name": "Distributed SMOGN",
#         "type": "dist",
#         "sampler": DistributedSMOGN,
#         "k_partitions": [2, 4, 8]
#     },
# }
#
SPARK = SparkSession.builder.master('local[4]').appName('Distributed Resampling').getOrCreate()
#
# execution_times = {}
#
# # adding debugging print statement
#
# # adding exception handling for all the sampling techniques
#
# #executing for 10 iterations - do this later
# for iteration in range(1):
#
#     for dataset, label_col in DATASETS.items():
#
#         print(dataset)
#         DATA_PROCESSED_TRAIN_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/train"
#         DATA_PROCESSED_TEST_DIR = f"{DATA_PROCESSED_DIR}/{dataset}/test"
#
#         print("Reading DF")
#         df = pd.read_csv(f"{DATA_RAW_DIR}/{dataset}.csv")
#
#         print("Reading DF in spark!")
#         df = spark.createDataFrame(df)
#
#         print("Calculating Phi value")
#         relevance_col = "phi"
#         df = Phi(input_col=label_col, output_col=relevance_col).transform(df)
#
#         print("Splitting and Pre-processing")
#         X = df.toPandas()
#         y = X[label_col]
#         k_fold_obj = KFold(n_splits=5, shuffle=True, random_state=42)
#
#         k_fold_splits = k_fold_obj.split(X, y)
#
#
#         for fold, (train_indices, test_indices) in enumerate(k_fold_splits):
#             print(f"Fold {fold}:")
#
#             X_train = X.iloc[train_indices]
#             X_test = X.iloc[test_indices]
#
#             # Remove index column
#             X_train.drop(columns=[relevance_col], inplace=True)
#             if "Unnamed: 0" in X_train.columns:
#                 X_train.drop(columns=["Unnamed: 0"], inplace=True)
#             if "Unnamed: 0" in X_test.columns:
#                 X_test.drop(columns=["Unnamed: 0"], inplace=True)
#
#             train = spark.createDataFrame(pd.DataFrame(X_train))
#             phi = X_test.pop(relevance_col)
#
#
#             print("Saving the CSV files")
#             X_test.to_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_Fold_{fold}.csv", index=False)
#             phi.to_csv(f"{DATA_PROCESSED_TEST_DIR}/{dataset}_phi_Fold_{fold}.csv", index=False)
#             X_train.to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_Fold_{fold}.csv", index=False)
#
#             # initialize for all folds in the first iteration ---> IMP
#             # initialize dataset time set for first iteration and first fold only ---> IMP
#             if fold == 0:
#                 execution_times[dataset] = {}
#             execution_times[dataset]["RUS_Fold " + str(fold)] = 0
#             execution_times[dataset]["ROS_Fold " + str(fold)] = 0
#             execution_times[dataset]["SMOGN_Fold " + str(fold)] = 0
#             execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold " + str(fold)] = 0
#             execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold " + str(fold)] = 0
#             execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold " + str(fold)] = 0
#
#         # execution_times[dataset] = {}
#
#             # try:
#             #     print("Initializing Distributed RUS")
#             #     start_time = time.time()
#             #     train_rus = DistributedRUS(label_col=label_col, k_partitions=1).transform(train)
#             #     end_time = time.time()
#             #     execution_times[dataset]["RUS_Fold " + str(fold)] = execution_times[dataset]["RUS_Fold " + str(fold)] + round(
#             #         end_time - start_time, 3)
#             #     train_rus.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_rus_Fold_{fold}.csv", index=False)
#             # except Exception as e:
#             #     print(f"Exception found in Distributed RUS: {e}")
#             #
#             # try:
#             #     print("Initializing Distributed ROS")
#             #     start_time = time.time()
#             #     train_ros = DistributedROS(label_col=label_col, k_partitions=1).transform(train)
#             #     end_time = time.time()
#             #     execution_times[dataset]["ROS_Fold " + str(fold)] = execution_times[dataset]["ROS_Fold " + str(fold)] + round(
#             #         end_time - start_time, 3)
#             #     train_ros.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_ros_Fold_{fold}.csv", index=False)
#             # except Exception as e:
#             #     print(f"Exception found in Distributed ROS: {e}")
#
#             # try:
#             #     print("Initializing SMOGN")
#             #     start_time = time.time()
#             #     train_smogn = smoter(data=train.toPandas(), y=label_col)
#             #     end_time = time.time()
#             #     execution_times[dataset]["SMOGN_Fold " + str(fold)] = execution_times[dataset]["SMOGN_Fold " + str(fold)] + round(
#             #         end_time - start_time, 3)
#             #     train_smogn.to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_smogn_Fold_{fold}.csv", index=False)
#             # except Exception as e:
#             #     print(f"Exception found in SMOGN: {e}")
#
#             # try:
#             #     print("Initializing Distributed SMOGN with 2 partitions!")
#             #     start_time = time.time()
#             #     train_dist_smogn_2 = DistributedSMOGN(label_col=label_col, k_partitions=2).transform(train)
#             #     end_time = time.time()
#             #     execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold " + str(fold)] = \
#             #     execution_times[dataset]["Distributed SMOGN (k_partitions = 2)_Fold " + str(fold)] + round(
#             #         end_time - start_time, 3)
#             #     train_dist_smogn_2.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_2_Fold_{fold}.csv",
#             #                                          index=False)
#             # except Exception as e:
#             #     print(f"Found exception in DistSMOGN-2P: {e}")
#             #
#             # try:
#             #     print("Initializing Distributed SMOGN with 4 partitions!")
#             #     start_time = time.time()
#             #     train_dist_smogn_4 = DistributedSMOGN(label_col=label_col, k_partitions=4).transform(train)
#             #     end_time = time.time()
#             #     execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold " + str(fold)] = \
#             #     execution_times[dataset]["Distributed SMOGN (k_partitions = 4)_Fold " + str(fold)] + round(
#             #         end_time - start_time, 3)
#             #     train_dist_smogn_4.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_4_Fold_{fold}.csv",
#             #                                          index=False)
#             # except Exception as e:
#             #     print(f"Found exception in DistSMOGN-4P: {e}")
#
#             try:
#                 print("Initializing Distributed SMOGN with 8 partitions!")
#                 start_time = time.time()
#                 train_dist_smogn_8 = DistributedSMOGN(label_col=label_col, k_partitions=8).transform(train)
#                 end_time = time.time()
#                 execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold " + str(fold)] = \
#                 execution_times[dataset]["Distributed SMOGN (k_partitions = 8)_Fold " + str(fold)] + round(
#                     end_time - start_time, 3)
#                 train_dist_smogn_8.toPandas().to_csv(f"{DATA_PROCESSED_TRAIN_DIR}/{dataset}_dist_smogn_8_Fold_{fold}.csv",
#                                                      index=False)
#             except Exception as e:
#                 print(f"Found exception in Dist-SMOGN-8P: {e}")
#
#     pd.DataFrame(data=execution_times).to_csv(f"{RESULT_EXECUTION_TIME_DIR}/execution_time_itr_{iteration}.csv", index=True)

def run_dist_rus(label_col, train):
    try:
        print("Initializing Distributed RUS")
        start_time = time.time()
        train_rus = DistributedRUS(label_col=label_col, k_partitions=1).transform(train)
        end_time = time.time()

        time_taken = round(end_time-start_time, 3)

        return train_rus.toPandas(), time_taken
    except Exception as e:
        print(f"Exception found in Distributed RUS: {e}")


def run_dist_ros(label_col, train):
    try:
        print("Initializing Distributed ROS")
        start_time = time.time()
        train_ros = DistributedROS(label_col=label_col, k_partitions=1).transform(train)
        end_time = time.time()

        time_taken = round(end_time - start_time, 3)

        return train_ros.toPandas(), time_taken

    except Exception as e:
        print(f"Exception found in Distributed ROS: {e}")


def run_smogn(label_col, train):
    try:
        print("Initializing SMOGN")
        start_time = time.time()
        train_smogn = smoter(data=train.toPandas(), y=label_col)
        end_time = time.time()

        time_taken = round(end_time - start_time, 3)

        return train_smogn, time_taken

    except Exception as e:
        print(f"Exception found in SMOGN: {e}")


def run_smogn_with_n_partitions(n, label_col, train):
    print(f"Running SMOGN with {n} partitions")
    try:
        start_time = time.time()
        train_dist_smogn_n = DistributedSMOGN(label_col=label_col, k_partitions=n).transform(train)
        end_time = time.time()
        time_taken = round(end_time - start_time, 3)

        return train_dist_smogn_n.toPandas(), time_taken

    except Exception as e:
        print(f"Found exception in DistSMOGN-{n}P: {e}")


def run_folds(dataset_name, label_col, random_state, iteration):
    print(f"Running {NUM_FOLDS} folds for {dataset_name}")

    # read dataset, push to spark and calculate Phi #########
    df = pd.read_csv(f"{DATA_RAW_DIR}/{dataset_name}.csv")

    print("Reading DF in spark!")
    df = SPARK.createDataFrame(df)

    print("Calculating Phi value")
    relevance_col = "phi"
    df = Phi(input_col=label_col, output_col=relevance_col).transform(df)

    ##########################################################

    # K Fold splitting here
    X = df.toPandas()
    y = X[label_col]
    k_fold_obj = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=random_state)

    k_fold_splits = k_fold_obj.split(X, y)

    for fold, (train_indices, test_indices) in enumerate(k_fold_splits):

        print(f"Performing fold {fold} for {dataset_name}")
        EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"] = {}

        X_train = X.iloc[train_indices]
        X_test = X.iloc[test_indices]

        X_train.drop(columns=[relevance_col], inplace=True)
        phi = X_test.pop(relevance_col)

        train = SPARK.createDataFrame(pd.DataFrame(X_train))

        # Save the dataframes so that you can perform No Sampling Experiments
        print("Saving the CSV files")
        X_train.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_iter_{iteration}_fold_{fold}.csv", index=False)
        X_test.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/test/{dataset_name}_iter_{iteration}_fold_{fold}.csv", index=False)
        phi.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/test/{dataset_name}_phi_iter_{iteration}_fold_{fold}.csv", index=False)

        # run sampling techniques #######################################

        # try:
        #     dist_rus_data, dist_rus_time = run_dist_rus(label_col, train)
        #     dist_rus_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_rus_iter_{iteration}_fold_{fold}.csv", index=False)
        #     EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["RUS"] = dist_rus_time
        # except Exception as e:
        #     print(f"Exception in RUS: {e}")
        #
        # try:
        #     dist_ros_data, dist_ros_time = run_dist_ros(label_col, train)
        #     dist_ros_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_ros_iter_{iteration}_fold_{fold}.csv", index=False)
        #     EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["ROS"] = dist_ros_time
        # except Exception as e:
        #     print(f"Exception in ROS: {e}")

        try:
            smogn_data, smogn_time = run_smogn(label_col, train)
            smogn_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_smogn_iter_{iteration}_fold_{fold}.csv", index=False)
            EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["SMOGN"] = smogn_time
        except Exception as e:
            print(f"Exception in SMOGN: {e}")

        # try:
        #     dist_smogn_2_data, dist_smogn_2_time = run_smogn_with_n_partitions(2, label_col, train)
        #     dist_smogn_2_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_distsmogn2_iter_{iteration}_fold_{fold}.csv", index=False)
        #     EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["Dist_SMOGN_2P"] = dist_smogn_2_time
        # except Exception as e:
        #     print(f"Exception in DIST SMOGN2: {e}")
        #
        # try:
        #     dist_smogn_4_data, dist_smogn_4_time = run_smogn_with_n_partitions(4, label_col, train)
        #     dist_smogn_4_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_distsmogn4_iter_{iteration}_fold_{fold}.csv", index=False)
        #     EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["Dist_SMOGN_4P"] = dist_smogn_4_time
        # except Exception as e:
        #     print(f"Exception in DIST SMOGN4: {e}")
        #
        # try:
        #     dist_smogn_8_data, dist_smogn_8_time = run_smogn_with_n_partitions(8, label_col, train)
        #     dist_smogn_8_data.to_csv(f"{DATA_PROCESSED_DIR}/{dataset_name}/train/{dataset_name}_distsmogn8_iter_{iteration}_fold_{fold}.csv", index=False)
        #     EXECUTION_TIME[f"iter_{iteration}_fold_{fold}"]["Dist_SMOGN_8P"] = dist_smogn_8_time
        # except Exception as e:
        #     print(f"Exception in DIST SMOGN8: {e}")


def run_iterations(dataset_name):

    print(f"Running {NUM_ITERATIONS} iterations for {dataset_name}")

    label_col = DATASET_TO_LABEL_MAP.get(dataset_name)

    for iteration in range(NUM_ITERATIONS):
        random_state = RANDOM_STATES[iteration]
        run_folds(dataset_name, label_col, random_state, iteration)


def orchestrate_resampling(dataset_name):
    run_iterations(dataset_name)
    pd.DataFrame(EXECUTION_TIME).to_csv(f"{RESULT_EXECUTION_TIME_DIR}/execution_time_{dataset_name}.csv", index=True)
    return


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dataset", type=str, required=True)

    arguments = arg_parser.parse_args()
    dataset_name = arguments.dataset

    orchestrate_resampling(dataset_name)
