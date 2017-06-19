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
