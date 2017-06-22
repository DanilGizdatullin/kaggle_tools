import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb


class CVSklearn(object):
    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def run_cv(self, df, pipeline_train, pipeline_test, pipeline_both, model, metrics):

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
            y_pred = model.predict(x_test)

            for i in range(len(metrics)):
                metrics_values_for_fold[0, i] = metrics[i](y_test, y_pred)
            metrics_values.append(metrics_values_for_fold)
        return np.vstack(metrics_values)


class CVXGBoost(object):
    def __init__(self, n_folds=5):
        self.n_folds = n_folds

    def run_cv(self, df, pipeline_train, pipeline_test, pipeline_both, metrics, dict_of_xgb):

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
