class EvaluationMethod:

    def __init__(self):
        self.no_groundtruth=False
        self.dir = None
        self.shape = None
        self.needs_sh = False


    def set_dir(self, dir):
        self.dir = dir
        

    def set_shape(self, shape):
        self.shape = shape


    def evaluate_both(self, i, LR, SR, HR):
        pass


    def evaluate_SR(self, i, LR, SR):
        pass

    
    def evaluate_SR_sh(self, i, LR, SR, SR_sh):
        pass


    def finalize(self):
        pass


    def summarize(self, paths, outdir):
        pass
