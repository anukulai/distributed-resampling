# distributed-resampling techniques for imbalanced regression problem

The different resampling techniques present in this repository are:
1) SMOGN
2) DistSMOGN
3) FastDistSMOGN
4) ADASYNR
5) FastDistADASYNR
6) RUS
7) ROS

## Prerequisites
1) Check the requirements.txt file and install all dependencies required to run the project. 
2) Make sure the datasets are present in `new_data/raw` folder.
3) Make sure `new_data/processed` has all the dataset folders and each such folder must have `train` and `test` sub-folders. 
4) Make sure the `new_results` folder is present to save all the results.
5) Make sure `new_results/predictive_performance` has folders with all dataset names ready before running.

## To Run FastDistSMOGN resampling (all partitions)
```bash
python /distributed-resampling/resampling_FastDistSMOGN.py --dataset {dataset-name}
```
## To Run FastDistADASYNR resampling (all partitions)
```bash
python /distributed-resampling/resampling_adasyn.py --dataset {dataset-name}
```

## To Run ADASYNR resampling
```bash
python /distributed-resampling/resampling_adasyn_base.py --dataset {dataset-name}
```

## To Run all other resampling techniques
```bash
python /distributed-resampling/resampling.py --dataset {dataset-name}
```

Once resampling is completed, we can then run the experiments on the processed data. There are 4 regressors used for the experiments namely:
1) Linear Regression (LR)
2) Support Vector Machine (SVM)
3) Random Forest (RF) 
4) Neural Network (NN)

The experiments will also produce results for base (no sampling) data apart from results for the above resampling techniques.

## To Run experiments for all techniques
```bash
python /distributed-resampling/experiments.py --dataset {dataset-name} --regressor {regressor}
```

`regressors : lr, svm, rf or nn`
