import pandas as pd

from unittest import TestCase
from pipeline.pipeline import Pipeline


class TestPipeline(TestCase):
    def test_run_pipeline(self):

        df = pd.DataFrame([[1, 1], [2, 2], [3, 3], [4, 4]], columns=['a', 'b'])

        def sum_columns(df):
            df['c'] = df['a'] + df['b']
            return df

        def multiply(df):
            df['c'] *= 2
            return df

        pipeline = Pipeline(pipeline=[sum_columns, multiply])
        print(df)

        new_df = pipeline.run_pipeline(df)
        print(new_df)
        self.assertTrue(new_df.equals(pd.DataFrame([[1, 1, 4], [2, 2, 8], [3, 3, 12], [4, 4, 16]],
                                                   columns=['a', 'b', 'c'])))
