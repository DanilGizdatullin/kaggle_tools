class Pipeline(object):
    """
    Pipeline for data transformations
    """
    def __init__(self, pipeline=None):
        """
        Initialize pipeline by using list of transformations functions.
        :param pipeline: list of transformations functions. Transformation function is a function with
        pd.DataFrame as the input and pd.DataFrame as the output
        :return:
        """
        self.pipeline = pipeline

    def add_transformation(self, function):
        """
        Add new transformation to pipeline
        :param function: function with pd.DataFrame as the input and pd.DataFrame as the output
        :return:
        """
        self.pipeline.append(function)

    def run_pipeline(self, df):
        """
        Run pipeline on some pd.DataFrame df
        :param df: pd.DataFrame
        :return: transformed pd.DataFrame
        """
        df_tr = df.copy()
        for func in self.pipeline:
            df_tr = func(df_tr)

        return df_tr


class PipelineTrainTest(object):
    """
    Pipeline for two datasets transformations
    """
    def __init__(self, pipeline=None):
        """
        Initialize pipeline by using list of transformations functions.
        :param pipeline: list of transformations functions. Transformation function is a function with two
        pd.DataFrames as the input and two pd.DataFrames as the output
        :return:
        """
        self.pipeline = pipeline

    def add_transformation(self, function):
        """
        Add new transformation to pipeline
        :param function: function with two pd.DataFrames as the input and two pd.DataFrames as the output
        :return:
        """
        self.pipeline.append(function)

    def run_pipeline(self, df1, df2):
        """
        :param df1: the first input pd.DataFrame
        :param df2: the second input pd.DataFrame
        :return: two transformed pd.DataFrame
        """
        df1_tr = df1.copy()
        df2_tr = df2.copy()
        for func in self.pipeline:
            df1_tr, df2_tr = func(df1_tr, df2_tr)

        return df1_tr, df2_tr
