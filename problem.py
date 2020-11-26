import json
import os
import typing
from dataclasses import dataclass
from os.path import join as pjoin

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types import BaseScoreType
from rampwf.workflows import Estimator
from sklearn.model_selection import GroupShuffleSplit

DATA_HOME = "data"
RANDOM_STATE = 777
rng = np.random.RandomState(RANDOM_STATE)

# --------------------------------------
# 0) Utils to manipulate data
# --------------------------------------


@dataclass
class WalkSignal:
    """Wrapper class around a numpy array containing a walk signal (with metadata)"""
    trial_code: str
    age: int
    gender: str
    height: float
    weight: int
    bmi: float
    laterality: str
    sensor: str
    pathology_group: str
    is_control: str
    foot: str  # left or right
    signal: typing.Any  # numpy array or pandas dataframe

    @classmethod
    def load_from_file(cls, code, data_home=DATA_HOME):
        fname = pjoin(data_home, code)
        with open(fname + ".json", "r") as file_handle:
            metadata = json.load(file_handle)
        signal = pd.read_csv(fname + ".csv", sep=",")  # left and right feet

        left_foot_cols = ["LAV", "LAX", "LAY",
                          "LAZ", "LRV", "LRX", "LRY", "LRZ"]
        right_foot_cols = ["RAV", "RAX", "RAY",
                           "RAZ", "RRV", "RRX", "RRY", "RRZ"]

        left_foot = cls(trial_code=code,
                        age=metadata["Age"],
                        gender=metadata["Gender"],
                        height=metadata["Height"],
                        weight=metadata["Weight"],
                        bmi=metadata["BMI"],
                        laterality=metadata["Laterality"],
                        sensor=metadata["Sensor"],
                        pathology_group=metadata["PathologyGroup"],
                        is_control=metadata["IsControl"],
                        foot="Left",
                        signal=signal[left_foot_cols].rename(columns=lambda name: name[1:]))
        right_foot = cls(trial_code=code,
                         age=metadata["Age"],
                         gender=metadata["Gender"],
                         height=metadata["Height"],
                         weight=metadata["Weight"],
                         bmi=metadata["BMI"],
                         laterality=metadata["Laterality"],
                         sensor=metadata["Sensor"],
                         pathology_group=metadata["PathologyGroup"],
                         is_control=metadata["IsControl"],
                         foot="Right",
                         signal=signal[right_foot_cols].rename(columns=lambda name: name[1:]))

        return left_foot, right_foot


def load_steps(code, data_home=DATA_HOME):
    """Return two lists of steps (left and right feet).

    Arguments:
        code {str} -- code of the trial, e.g. "2-10"

    Keyword Arguments:
        data_home {str} -- folder where the files lie (default: {DATA_HOME})

    Returns:
        [tuple(list)] -- two lists of steps (left foot, right foot)
    """
    fname = pjoin(data_home, code)
    with open(fname + ".json", "r") as file_handle:
        metadata = json.load(file_handle)
    return metadata["LeftFootActivity"], metadata["RightFootActivity"]


def _read_data(path, train_or_test="train"):
    """Return the list of signals and steps for the train or test data set

    Arguments:
        path {str} -- folder where the train and test data are

    Keyword Arguments:
        train_or_test {str} -- train or test (default: {"train"})

    Returns:
        [tupe(List[WalkSignal], List)] -- (list of signals, list of lists of steps)
    """
    folder = pjoin(path, DATA_HOME, train_or_test)
    code_list = [fname.split(".")[0] for fname in os.listdir(
        folder) if fname.endswith(".csv")]

    test = os.getenv('RAMP_TEST_MODE', 0)  # are we in test mode
    if test:
        code_sublist = rng.choice(code_list, 5, replace=False)
    else:
        code_sublist = code_list

    X = list()
    y = list()
    for code in code_sublist:
        left_signal, right_signal = WalkSignal.load_from_file(code, folder)
        left_steps, right_steps = load_steps(code, folder)
        X.extend((left_signal, right_signal))
        y.extend((left_steps, right_steps))

    return X, np.array(y, dtype=list)


# --------------------------------------
# 2) Object implementing the score type
# --------------------------------------


def _check_step_list(step_list):
    """Some sanity checks."""
    for step in step_list:
        assert len(
            step) == 2, f"A step consists of a start and an end: {step}."
        start, end = step
        assert start < end, f"start should be before end: {step}."


def _step_detection_precision(step_list_true, step_list_pred):
    """Precision is the number of correctly predicted steps divided by the number of predicted
    steps. A predicted step is counted as correct if its mid-index (mean of its start and end
    indexes) lies inside an annotated step.
    Note that an annotated step can only be detected once. If several predicted steps correspond
    to the same annotated step, all but one are considered as false.
    Here, precision is computed on a single prediction task (all steps correspond to the same
    signal).

    The lists y_true_ and y_pred are lists of steps, for instance:
        - step_list_true: [[357, 431], [502, 569], [633, 715], [778, 849], [907, 989]]
        - step_list_pred: [[293, 365], [422, 508], [565, 642], [701, 789]]

    Arguments:
        step_list_true {List} -- list of true steps
        step_list_pred {List} -- list of predicted steps

    Returns:
        float -- precision, between 0.0 and 1.0
    """
    _check_step_list(step_list_true)
    _check_step_list(step_list_pred)

    if len(step_list_pred) == 0:  # empty prediction
        return 0.0

    n_correctly_predicted = 0
    detected_index_set = set()  # set of index of detected true steps
    for (start_pred, end_pred) in step_list_pred:
        mid = (start_pred + end_pred) // 2
        for (index, (start_true, end_true)) in enumerate(step_list_true):
            if (index not in detected_index_set) and (start_true <= mid < end_true):
                n_correctly_predicted += 1
                detected_index_set.add(index)
                break

    return n_correctly_predicted / len(step_list_pred)


def _step_detection_recall(step_list_true, step_list_pred):
    """Recall is the number of detected annotated steps divided by the total number of annotated
    steps. An annotated step is counted as detected if its mid-index lies inside a predicted step.
    Note that an annotated step can only be detected once. If several annotated steps are detected
    with the same predicted step, all but one are considered undetected.
    Here, recall is computed on a single prediction task (all steps correspond to the same
    signal).

    The lists y_true_ and y_pred are lists of steps, for instance:
        - step_list_true: [[357, 431], [502, 569], [633, 715], [778, 849], [907, 989]]
        - step_list_pred: [[293, 365], [422, 508], [565, 642], [701, 789]]

    Arguments:
        step_list_true {List} -- list of true steps
        step_list_pred {List} -- list of predicted steps

    Returns:
        float -- recall, between 0.0 and 1.0
    """
    _check_step_list(step_list_true)
    _check_step_list(step_list_pred)

    n_detected_true = 0
    predicted_index_set = set()  # set of indexes of predicted steps

    for (start_true, end_true) in step_list_true:
        mid = (start_true + end_true) // 2
        for (index, (start_pred, end_pred)) in enumerate(step_list_pred):
            if (index not in predicted_index_set) and (start_pred <= mid < end_pred):
                n_detected_true += 1
                predicted_index_set.add(index)
                break
    return n_detected_true / len(step_list_true)


class FScoreStepDetection(BaseScoreType):
    is_lower_the_better = False
    minimum = 0.0
    maximum = 1.0

    def __init__(self, name="F-score (step detection)", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred) -> float:
        """
        Calculate f-score (geometric mean between precision and recall) for each instance (each
        signal) and return the weighted average over instances.

        The lists y_true_ and y_pred are lists of lists of steps, for instance:
            - y_true: [[[907, 989]] [[357, 431], [502, 569]], [[633, 715], [778, 849]]]
            - y_pred: [[[293, 365]], [[422, 508], [565, 642]], [[701, 789]]]

        Arguments:
            y_true {List} -- true steps
            y_pred {List} -- predicted steps

        Returns:
            float -- f-score, between 0.0 and 1.0
        """
        fscore_list = list()

        print("List size :")
        print(len(y_true))

        for (step_list_true, step_list_pred) in zip(y_true, y_pred):
            prec = _step_detection_precision(step_list_true, step_list_pred)
            rec = _step_detection_recall(step_list_true, step_list_pred)
            if prec + rec < 1e-6:
                fscore_list.append(0.0)
            else:
                score = (2 * prec * rec) / (prec + rec)
                fscore_list.append(score)
                if score < 1:
                    ax = pd.DataFrame(np.arange(0, step_list_true[-1][1] + 20)).plot(figsize=(20, 10))
                    line_args = {"linestyle": "--", "color": "k"}
                    for (start, end) in step_list_true:
                        ax.axvline(start, **line_args)
                        ax.axvline(end, **line_args)
                        ax.axvspan(start, end, facecolor='g', alpha=0.3)
                        
                    for (start, end) in step_list_pred:
                        ax.axvline(start, **line_args)
                        ax.axvline(end, **line_args)
                        ax.axvspan(start, end, facecolor='r', alpha=0.3)
                    plt.show()

        return np.mean(fscore_list)

# --------------------------------------
# 3) Prediction types
# --------------------------------------


class _Predictions(BasePrediction):

    def __init__(self, y_pred=None, y_true=None, n_samples=None):
        """Essentially the same as in a regression task, but the prediction is a list not a float."""
        if y_pred is not None:
            self.y_pred = np.array(y_pred, dtype=list)
        elif y_true is not None:
            self.y_pred = np.array(y_true, dtype=list)
        elif n_samples is not None:
            # self.n_columns == 0:
            shape = (n_samples)
            self.y_pred = np.empty(shape, dtype=list)
            self.y_pred.fill(np.nan)
        else:
            raise ValueError(
                'Missing init argument: y_pred, y_true, or n_samples')
        self.check_y_pred_dimensions()

    @property
    def valid_indexes(self):
        """Return valid indices (e.g., a cross-validation slice)."""
        if len(self.y_pred.shape) == 1:
            return ~pd.isnull(self.y_pred)
        elif len(self.y_pred.shape) == 2:
            return ~pd.isnull(self.y_pred[:, 0])
        else:
            raise ValueError('y_pred.shape > 2 is not implemented')

    def check_y_pred_dimensions(self):
        pass

    @classmethod
    def combine(cls, predictions_list, index_list=None):
        """Dummy function. Here, combining consists in taking the first prediction."""
        combined_predictions = cls(y_pred=predictions_list[0].y_pred)
        return combined_predictions


def make_step_detection():
    return _Predictions

# --------------------------------------
# 4) Ramp problem definition
# --------------------------------------


problem_title = "Step Detection with Inertial Measurement Units"
Predictions = make_step_detection()
workflow = Estimator()
score_types = [FScoreStepDetection(name="F-score (step detection)")]


def get_train_data(path="."):
    return _read_data(path, 'train')


def get_test_data(path="."):
    return _read_data(path, 'test')


def get_cv(X, y):
    """
    In this cross-validation scheme, for a single trial, the left and right signals are
    not in different folds and test/train sets, therefore the cross-validation is
    stratified according to the `trial_code` attribute.
    """
    cv = GroupShuffleSplit(
        n_splits=5, test_size=0.2, random_state=RANDOM_STATE)
    code_list = [signal.trial_code for signal in X]
    return cv.split(X, y, code_list)
