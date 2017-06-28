import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb


class CVSklearn(object):
    """
    This class helps to make cross-validation for classification or regression task by using model from scikit-learn
    """
    def __init__(self, n_folds=5):
        """
        Initialize object to create cross-validation
        :param n_folds: number of folds in cross-validation
        :return:
        """
        self.n_folds = n_folds

    def run_cv(self, df, pipeline_train, pipeline_test, pipeline_both, model, metrics, is_predict_proba=False):
        """
        Run cross-validation. The last column after pipeline_train, pipeline_test and pipeline_both is target column.
        :param df: pd.DataFrame with data for cv.
        :param pipeline_train: pipeline with transformations for train data
        :param pipeline_test: pipeline with transformations for test data
        :param pipeline_both: pipeline with transformations for both (train, test).
        It starts after pipeline_train and pipeline_test.
        :param model: model from sklearn with methods fit() and predict() or predict_proba()
        :param metrics: list of some metrics from sklearn.metrics
        :param is_predict_proba: if it's True model uses predict_proba() method otherwise predict() method
        :return: np.array with shape (self.n_folds, len(metrics)) where x is fold_id
        and y is correspondent metric from metrics
        """
        kf = KFold(n_splits=self.n_folds, shuffle=True)

        columns = df.columns

        x = df.values[:, :-1]
        y = df.values[:, -1]

        metrics_values = []

        for train, test in kf.split(x):

            metrics_values_for_fold = np.zeros((1, len(metrics)))

            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

            train_df = pd.DataFrame(x_train, columns=columns[0: -1])
            train_df[columns[-1]] = y_train
            train_df = pipeline_train.run_pipeline(train_df)

            test_df = pd.DataFrame(x_test, columns=columns[0: -1])
            test_df[columns[-1]] = y_test
            test_df = pipeline_test.run_pipeline(test_df)

            if pipeline_both:
                train_df, test_df = pipeline_both.run_pipeline((train_df, test_df))

            x_train, x_test = train_df.values[:, :-1], test_df.values[:, :-1]
            y_train, y_test = train_df.values[:, -1], test_df.values[:, -1]

            model.fit(x_train, y_train)
            if is_predict_proba:
                y_pred = model.predict_proba(x_test)[:, 1]
            else:
                y_pred = model.predict(x_test)

            for i in range(len(metrics)):
                metrics_values_for_fold[0, i] = metrics[i](y_test, y_pred)
            metrics_values.append(metrics_values_for_fold)
        return np.vstack(metrics_values)


class CVXGBoost(object):
    """
    This class helps to make cross-validation for classification or regression task by using model from XGBoost
    """
    def __init__(self, n_folds=5):
        """
        Initialize object to create cross-validation
        :param n_folds: number of folds in cross-validation
        :return:
        """
        self.n_folds = n_folds

    def run_cv(self, df, pipeline_train, pipeline_test, pipeline_both, metrics, dict_of_xgb):
        """
        Run cross-validation. The last column after pipeline_train, pipeline_test and pipeline_both is target column.
        :param df: pd.DataFrame with data for cv
        :param pipeline_train: pipeline with transformations for train data
        :param pipeline_test: pipeline with transformations for test data
        :param pipeline_both: pipeline with transformations for both (train, test).
        It starts after pipeline_train and pipeline_test.
        :param metrics: list of some metrics from sklearn.metrics
        :param dict_of_xgb: dict with key 'xgb_params' and value dict with parameters for XGBoost
        and key 'num_boost_round' and value N where N - is number of boosting rounds
        :return: np.array with shape (self.n_folds, len(metrics)) where x is fold_id
        and y is correspondent metric from metrics
        """

        kf = KFold(n_splits=self.n_folds, shuffle=True)

        columns = df.columns

        x = df.values[:, :-1]
        y = df.values[:, -1]

        metrics_values = []

        for train, test in kf.split(x):

            metrics_values_for_fold = np.zeros((1, len(metrics)))

            x_train, x_test, y_train, y_test = x[train], x[test], y[train], y[test]

            train_df = pd.DataFrame(x_train, columns=columns[0: -1])
            train_df[columns[-1]] = y_train
            train_df = pipeline_train.run_pipeline(train_df)

            test_df = pd.DataFrame(x_test, columns=columns[0: -1])
            test_df[columns[-1]] = y_test
            test_df = pipeline_test.run_pipeline(test_df)

            if pipeline_both:
                train_df, test_df = pipeline_both.run_pipeline(train_df, test_df)

            x_train, x_test = train_df.values[:, :-1], test_df.values[:, :-1]
            y_train, y_test = train_df.values[:, -1], test_df.values[:, -1]

            dtrain = xgb.DMatrix(x_train, y_train, missing=np.NaN)
            dtest = xgb.DMatrix(x_test, missing=np.NaN)

            xgb_params = dict_of_xgb['xgb_params']
            num_boost_round = dict_of_xgb['num_boost_round']

            model = xgb.train(dict(xgb_params, silent=1), dtrain, num_boost_round=num_boost_round,
                              verbose_eval=True)
            y_pred = model.predict(dtest)

            for i in range(len(metrics)):
                metrics_values_for_fold[0, i] = metrics[i](y_test, y_pred)
            metrics_values.append(metrics_values_for_fold)
        return np.vstack(metrics_values)
