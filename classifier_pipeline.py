"""
Script that performs grid search for all classifiers across various datasets.
Datasets are balanced via KMeansSMOTE and results are obtained via stratified
 10-fold cross-validation.
"""
import argparse
import pickle
from pathlib import Path
from typing import Union

import optuna
import tqdm
import numpy as np
import pandas as pd

from imblearn.metrics import geometric_mean_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, recall_score, matthews_corrcoef, balanced_accuracy_score, make_scorer

from classifier_configs import get_classifier, grids
from src.checksum import update_checksums

import os
import warnings
warnings.simplefilter("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

scoring_dict = {
    "mcc": make_scorer(matthews_corrcoef),
    "accuracy": make_scorer(accuracy_score),
    "recall": make_scorer(recall_score),
    "specificity": make_scorer(recall_score, pos_label=0),
    "gm": make_scorer(geometric_mean_score),
    "uar": make_scorer(balanced_accuracy_score, adjusted=False),
    "bm": make_scorer(balanced_accuracy_score, adjusted=True)
}

# Choose the main metric to optimize
metric_to_optimize = "matthews_corrcoef"

def get_datasets_to_process(datasets_path: Path, results_data: Path, dataset_slice: Union[tuple, int, None] = None):
    td = sorted([str(x.name) for x in datasets_path.iterdir()])
    if isinstance(dataset_slice, tuple):
        td = list(td)[dataset_slice[0]:dataset_slice[1]]
    elif isinstance(dataset_slice, int):
        td = list(td)[:dataset_slice]
    tr = sorted([str(x.name) for x in results_data.iterdir()])
    to_do = sorted(list(set(td) - set(tr)))
    return to_do

def main(sex: str = "women", classifier="svm_poly", dataset_slice=None):
    training_data = Path(".").joinpath("training_data", sex)
    results_data = Path(".").joinpath("results", classifier, sex)
    results_data.mkdir(exist_ok=True, parents=True)

    dataset = get_datasets_to_process(training_data, results_data, dataset_slice)
    dataset = sorted(dataset)

    for training_dataset_str in tqdm.tqdm(dataset):
        results_file = results_data.joinpath(str(training_dataset_str))
        training_dataset = training_data.joinpath(training_dataset_str)
        results_data.joinpath(str(training_dataset.name)).mkdir(parents=True, exist_ok=True)

        with open(training_data.joinpath(str(training_dataset.name), "dataset.pk"), "rb") as f:
            train_set = pickle.load(f)
        data_X = np.array(train_set["data"], dtype=np.float64)
        data_y = np.array(train_set["labels"])
        trial_results = []
        def objective(trial):
            pipeline, param_grid = get_classifier(classifier, both_sexes=(sex == "both"), random_seed=RANDOM_SEED)
            trial_params = {}
            for param_name, param_values in grids[classifier].items():
                if all(isinstance(v, (int, float)) for v in param_values):
                    if all(isinstance(v, int) for v in param_values):
                        val = trial.suggest_int(param_name, min(param_values), max(param_values))
                    else:
                        val = trial.suggest_float(param_name, min(param_values), max(param_values), log=True)
                else:
                    val = trial.suggest_categorical(param_name, param_values)
                pipeline.set_params(**{param_name: val})
                trial_params[param_name] = val

            scores = cross_val_score(pipeline, data_X, data_y,
                                     cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED),
                                     scoring=metric_to_optimize, n_jobs=-1)
            trial_score = scores.mean()
            trial_results.append({**trial_params, f"mean_test_{metric_to_optimize}": trial_score})
            return trial_score

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=50)

        results_file.mkdir(exist_ok=True)
        header = not results_file.joinpath("results.csv").exists()

        df_result = pd.DataFrame(trial_results)
        df_result.to_csv(results_file.joinpath("results.csv"), index=False, mode="a", header=header, encoding="utf8",
                         lineterminator="\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="classifier_pipeline.py",
        description="Perform Optuna optimization for different classifiers across various datasets."
    )
    parser.add_argument("classifier", type=str)
    parser.add_argument("sex", type=str)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    sex_to_compute = args.sex
    used_classifier = args.classifier
    data_slice = None if not args.test else 50
    main(sex_to_compute, used_classifier, data_slice)
    update_checksums(Path("results"), Path("misc").joinpath("after_IV.sha256"))
