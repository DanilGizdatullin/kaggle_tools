import pandas as pd
from unittest import TestCase
from pipeline.cv import CVSklearn, CVXGBoost
from pipeline.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def add_sum(df):
    df['a12'] = df['a1'] + df['a2']
    df = df[['a1', 'a2', 'a3', 'a4', 'a12', 'target']]
    return df


class TestCV(TestCase):
    def test_run_cv_sklearn(self):
        data = load_iris()
        df = pd.DataFrame(data.data, columns=['a1', 'a2', 'a3', 'a4'])
        df['target'] = data.target
        model = LogisticRegression()
        pipeline_train = Pipeline([add_sum])
        pipeline_test = Pipeline([add_sum])

        cv = CVSklearn(n_folds=5)
        results = cv.run_cv(df, pipeline_train, pipeline_test, None, model, metrics=[accuracy_score])

        self.assertTrue(results.shape == (5, 1))

    def test_run_cv_xgb(self):
        data = load_iris()
        df = pd.DataFrame(data.data, columns=['a1', 'a2', 'a3', 'a4'])
        df['target'] = data.target
        pipeline_train = Pipeline([add_sum])
        pipeline_test = Pipeline([add_sum])

        dict_of_xgb = dict()
        dict_of_xgb['xgb_params'] = {
            'eta': 0.05,
            'max_depth': 6,
            'subsample': 0.6,
            'colsample_bytree': 1,
            'objective': 'multi:softmax',
            'eval_metric': 'logloss',
            'silent': 1,
            'num_class': 3
        }
        dict_of_xgb['num_boost_round'] = 400

        cv = CVXGBoost(n_folds=5)
        results = cv.run_cv(df, pipeline_train, pipeline_test, None, metrics=[accuracy_score], dict_of_xgb=dict_of_xgb)

        self.assertTrue(results.shape == (5, 1))
