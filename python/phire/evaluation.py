import numpy as np
from glob import glob
import pyshtools as pysh

from .PhIREGANs import PhIREGANs


class PowerSpectrum:

    def __init__(self):
        self.spectrums_SR = []
        self.spectrums_HR = []

    
    def evaluate(self, batch_LR, batch_SR, batch_HR):
        pass

    
    def finalize(self):
        pass


class Evaluation:

    def __init__(self):
        self.dataset = sorted(glob('/data2/stengel/whole_images/stengel_eval_1995_1999.*.tfrecords'))    
        self.checkpoint = 'models/resnet-20210525-114614/pretraining/generator-30'

        ##########################################

        self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187]
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]

        self.save_every = 1
        self.batch_size = 64
        self.r = 4

        self.metrics = {
            'power-spectrum': PowerSpectrum()
        }

    
    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.exp1m(np.fabs(y))
        return y*self.std + self.mean


    def run(self):
        gan = PhIREGANs('eval', mu_sig=[[0,0], [1,1]], print_every=40)

        iter_ = gan.test(
            self.r, 
            self.dataset, 
            self.checkpoint, 
            batch_size=self.batch_size, 
            save_every=self.save_every, 
            return_batches=True, 
            return_hr=True
        )

        for i, (batch_LR, batch_SR, batch_HR) in enumerate(iter_):
            batch_LR = self.deprocess(batch_LR)
            batch_SR = self.deprocess(batch_SR)
            batch_HR = self.deprocess(batch_HR)

            for metric in self.metrics.values():
                metric.evaluate(batch_LR, batch_SR, batch_HR)

        for metric in self.metrics.values():
            metric.finalize()


def main():
    Evaluation().run()


if __name__ == '__main__':
    main()