class Pipeline(object):
    def __init__(self, pipeline=[]):
        self.pipeline = pipeline

    def add_transformation(self, function):
        self.pipeline.append(function)

    def run_pipeline(self, df):
        df_tr = df.copy()
        for func in self.pipeline:
            df_tr = func(df_tr)

        return df_tr


class PipelineTrainTest(Pipeline):
    def run_pipeline(self, df):
        df1 = df[0]
        df2 = df[1]
        df1_tr = df1.copy()
        df2_tr = df2.copy()
        for func in self.pipeline:
            df1_tr, df2_tr = func(df1_tr, df2_tr)

        return df1_tr, df2_tr
