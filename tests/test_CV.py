import pandas as pd
from unittest import TestCase
from pipeline.cv import CV
from pipeline.pipeline import Pipeline
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def add_sum(df):
    df['a12'] = df['a1'] + df['a2']
    df = df[['a1', 'a2', 'a3', 'a4', 'a12', 'target']]
    return df


class TestCV(TestCase):
    def test_run_cv(self):
        data = load_iris()
        df = pd.DataFrame(data.data, columns=['a1', 'a2', 'a3', 'a4'])
        df['target'] = data.target
        model = LogisticRegression()
        pipeline_train = Pipeline([add_sum])
        pipeline_test = Pipeline([add_sum])

        cv = CV(n_folds=5)
        results = cv.run_cv(df, pipeline_train, pipeline_test, model, metrics=[accuracy_score])

        self.assertTrue(results.shape == (5, 1))
