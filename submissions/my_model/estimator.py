import numpy as np
from scipy.signal import argrelmax
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

RANDOM_STATE = 7
rng = np.random.RandomState(RANDOM_STATE)


class Detector(BaseEstimator):
    def __init__(self, threshold=0.5, order=40):
        self.threshold = threshold
        self.order = order
        self.step_template = None

    def fit(self, X, y):
        assert len(X) == len(y), f"Wrong dimensions (len(X): {len(X)}, len(y): {len(y)})."
        return self

    def predict(self, X):
        y_pred = list()
        for sensor_data in X:
            sensor_data.signal["A_abs_sum"] = (
                pd.DataFrame(MinMaxScaler()
                .fit_transform(sensor_data.signal[["AX", "AY", "AZ", "AV", "RX", "RY", "RZ", "RV"]]
                .abs()))
                .sum(axis=1)
                .rolling(25, center = True)
                .sum()
            )
            threshold = sensor_data.signal["A_abs_sum"].median()
            sensor_data.signal["is_step"] = sensor_data.signal[
                "A_abs_sum"
            ] >= threshold
            prediction = []
            beg_step = True
            cur_step = []
            for index, value in sensor_data.signal["is_step"].items():
                if beg_step:
                    if value == True:
                        cur_step.append(index)
                        beg_step = False
                else:
                    if value == False:
                        cur_step.append(index)
                        prediction.append(cur_step)
                        cur_step = []
                        beg_step = True
                        
            steps_len = np.array([pred[1] - pred[0] for pred in prediction])
            
            prediction = [
                pred
                for pred in prediction
                if pred[1]-pred[0] > np.median(steps_len) / 3
            ]

            y_pred.append(prediction)

        return np.array(y_pred, dtype=list)


def get_estimator():
    # step detection
    detector = Detector()

    # make pipeline
    pipeline = Pipeline(steps=[("detector", detector)])

    return pipeline
