"""
Optuna example that optimizes a classifier configuration for IMDB movie review dataset.
This script is based on the example of allentune (https://github.com/allenai/allentune).

In this example, we optimize the validation accuracy of
sentiment classification using an AllenNLP jsonnet config file.
Since it is too time-consuming to use the training dataset,
we here use the validation dataset instead.

"""

import os.path
import shutil

import optuna
from optuna.integration import AllenNLPExecutor
from optuna.integration.allennlp import dump_best_config
from packaging import version

import allennlp


# This path trick is used since this example is also
# run from the root of this repository by CI.
EXAMPLE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(EXAMPLE_DIR, "classifier_distributed.jsonnet")
MODEL_DIR = "result_multi"
BEST_CONFIG_PATH = "best_classifier_multi.json"


def objective(trial):
    trial.suggest_float("DROPOUT", 0.0, 1.0)
    trial.suggest_int("EMBEDDING_DIM", 16, 512)
    trial.suggest_int("MAX_FILTER_SIZE", 3, 6)
    trial.suggest_int("NUM_FILTERS", 16, 256)
    trial.suggest_int("HIDDEN_SIZE", 16, 256)

    serialization_dir = os.path.join(MODEL_DIR, "test_{}".format(trial.number))
    executor = AllenNLPExecutor(trial, CONFIG_PATH, serialization_dir, force=True)

    return executor.run()


if __name__ == "__main__":
    if version.parse(allennlp.__version__) < version.parse("2.0.0"):
        raise RuntimeError(
            "`allennlp>=2.0.0` is required for this example."
            " If you want to use `allennlp<2.0.0`, please install `optuna==2.5.0`"
            " and refer to the following example:"
            " https://github.com/optuna/optuna/blob/v2.5.0/examples/allennlp/allennlp_jsonnet.py"
        )

    study = optuna.create_study(
        direction="maximize",
        storage="sqlite:///allennlp.db",
        pruner=optuna.pruners.HyperbandPruner(),
        sampler=optuna.samplers.TPESampler(seed=10),
        study_name="allennlp_jsonnet_multi",
        load_if_exists=True,
    )

    # study.optimize(objective, n_trials=50, timeout=600)
    study.optimize(objective, n_trials=100)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    dump_best_config(CONFIG_PATH, BEST_CONFIG_PATH, study)
    print("\nCreated optimized AllenNLP config to `{}`.".format(BEST_CONFIG_PATH))

    shutil.rmtree(MODEL_DIR)
