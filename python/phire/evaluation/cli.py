import os
import sys

from numpy.lib.arraysetops import isin
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
from glob import glob
from pathlib import Path
from time import time
import pyshtools as pysh

from phire.PhIREGANs import PhIREGANs

from .visualize import Visualize
from .moments import Moments
from .semivariogram import Semivariogram
from .spectrum import PowerSpectrum


class GANIterator:

    def __init__(self, outdir, checkpoint):
        self.outdir = outdir
        self.checkpoint = checkpoint        
        self.r = [2,2]


    def __call__(self, dataset, mean, std):
        mu_sig = [mean, std] if mean and std else None
        gan = PhIREGANs('eval', mu_sig=mu_sig, print_every=1e9, compression='ZLIB')
        iter_ = gan.test(
            self.r, 
            dataset, 
            self.checkpoint, 
            batch_size=2, 
            save_every=1,  
            return_hr=True,
            only_hr=False
        )

        return iter_


class BilinearIterator:

    def __init__(self, outdir):
        self.outdir = outdir

    def __call__(self, dataset, mean, std):
        mu_sig = [mean, std] if mean and std else None
        gan = PhIREGANs('eval', mu_sig=mu_sig, print_every=1e9, compression='ZLIB')
        
        # find shapes
        _, tmp_LR, tmp_HR = next(gan.iterate_data(dataset, batch_size=1))

        LR_in = tf.placeholder(tf.float32, [None] + list(tmp_LR.shape[1:]))
        bilinear = tf.image.resize(LR_in, [tmp_HR.shape[1], tmp_HR.shape[2]], tf.image.ResizeMethod.BILINEAR)

        for idx, LR, HR in gan.iterate_data(dataset, batch_size=2):
            SR = tf.get_default_session().run(bilinear, feed_dict={LR_in: LR})
            yield LR, SR, HR


class GroundtruthIterator:

    def __init__(self, outdir):
        self.outdir = outdir

    def __call__(self, dataset, mean, std):
        mu_sig = [mean, std] if mean and std else None
        gan = PhIREGANs('eval', mu_sig, print_every=1e9, compression='ZLIB')
        for idx, LR, HR in gan.iterate_data(dataset, batch_size=2):
            SR = HR
            yield LR, SR, HR


class Evaluation:

    def __init__(self, iterator, force_overwrite=False):
        self.iterator = iterator
        self.force_overwrite=force_overwrite
        self.is_groundtruth = isinstance(self.iterator, GroundtruthIterator)

        self.measure_time = False

        self.dataset = sorted(glob('/data/sr/sr_eval_2000_2005.*.tfrecords'))        
        self.denorm = True

        self.mean, self.std = [1.9464334e-08, 2.0547947e-07], [2.8568757e-05, 5.081943e-05]  # 1979-1988 div vort
        self.mean_log1p, self.std_log1p = [0.0008821452, 0.00032483143], [0.15794525, 0.16044095]  # alpha = 0.2

        self.setup_metrics()
        self.test_dims()


    def setup_metrics(self):
        self.px_per_deg = 2560 / 360
        img_patch_size = (self.to_px(60), self.to_px(60))
        img_freq = 1000

        vis_sea = Visualize((self.to_px(60), self.to_px(90)), img_patch_size, img_freq)
        vis_eu = Visualize((self.to_px(40), self.to_px(0)), img_patch_size, img_freq)
        vis_na = Visualize((self.to_px(40), self.to_px(250)), img_patch_size, img_freq)
        vis_pac = Visualize((self.to_px(60), self.to_px(180)), img_patch_size, img_freq)

        if self.denorm:
            metrics = {
                'power-spectrum': PowerSpectrum(1280, 2560),
                'semivariogram': Semivariogram(n_samples=20),
                'moments': Moments(),
                'img-SEA': vis_sea,
                'img-EU': vis_eu,
                'img-NA': vis_na,
                'img-pacific': vis_pac,
            }
        else:
            metrics = {
                'img-SEA-transformed': vis_sea,
                'img-EU-transformed': vis_eu,
                'img-NA-transformed': vis_na,
                'img-pacific-transformed': vis_pac,
            }

        self.metrics = {k:metric for k, metric in metrics.items() if not metric.no_groundtruth or not self.is_groundtruth}


    def to_px(self, deg):
        return int(round(deg*self.px_per_deg))


    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.expm1(np.fabs(y)) / 0.2  # alpha=0.2
        return y*self.std + self.mean


    def create_dirs(self, dir):
        for name, metric in self.metrics.items():
            metric_dir = Path(dir) / name

            try:
                os.makedirs(metric_dir, exist_ok=self.force_overwrite)
            except OSError:
                resp = input(f'{metric_dir} already exists. Continue? (y/n)')
                if resp != 'y' and resp != 'Y':
                    print('Evaluation cancelled')
                    return False
                else:
                    os.makedirs(metric_dir, exist_ok=True)
            
            metric.set_dir(metric_dir)
        
        return True


    def test_dims(self):
        LR, SR, HR = next(self.iterator(self.dataset, None, None))

        if SR.shape != HR.shape:
            print(f'SR shape does not match HR shape: {SR.shape[1:]} vs {HR.shape[1:]}')
            sys.exit(1)

        if HR.shape[1] % 2 != 0 or HR.shape[2] % 2 != 0:
            print(f'[WARNING] HR shape is not divisible by 2: {HR.shape[1:]}')

        if LR.shape[1] % 2 != 0 or LR.shape[2] % 2 != 0:
            print(f'[WARNING] LR shape is not divisible by 2: {LR.shape[1:]}')

        if HR.shape[1] != 1280 or HR.shape[2] != 2560:
            print(f'[WARNING] HR shape is not 1280x2560: {HR.shape[1:]}')



    def run(self):
        calc_sh = any(metric.needs_sh for metric in self.metrics.values())

        if not self.create_dirs(self.iterator.outdir):
            return

        # data is already log1p normalized and z-normed
        iter_ = self.iterator(self.dataset, None, None)
        try:

            t1_gan = time()
            for i, (LR, SR, HR) in enumerate(iter_):
                if self.measure_time:
                    print(f'inference took {time()-t1_gan:.2f}s')
                    t1_gan = time()

                if self.denorm:
                    LR = self.deprocess(LR)
                    HR = self.deprocess(HR)
                    SR = self.deprocess(SR)
                
                if calc_sh:
                    t1 = time()
                    C = SR.shape[-1]
                    SR_sh = [np.stack([pysh.expand.SHExpandDH(img) for img in SR[..., c]], axis=0) for c in range(C)]
                    SR_sh = np.stack(SR_sh, axis=-1)
                    t2 = time()
                    
                    if self.measure_time:
                        print(f'sh-transform took {t2-t1:.2f}s')


                for name, metric in self.metrics.items():
                    if i == 0:
                        metric.set_shape(HR.shape[1:])

                    t1 = time()
                    if metric.needs_sh:
                        metric.evaluate_SR_sh(i, LR, SR, SR_sh)
                    else:
                        metric.evaluate_both(i, LR, SR, HR)
                        metric.evaluate_SR(i, LR, SR)
                    t2 = time()

                    if self.measure_time:
                        print(f'{name} took {t2-t1:.2f}s')

                print(i, flush=True)


            for metric in self.metrics.values():
                metric.finalize()

        finally:
            # make sure that any ressources (tf.Session) held by iter_ get released now!
            # this is important
            del iter_


    def summarize(self, outdir, paths):
        if not self.create_dirs(outdir):
            return

        for metric_name, metric in self.metrics.items():
            metric_paths = {name: Path(path) / metric_name for name,path in paths.items()}
            metric.summarize(metric_paths, metric.dir)



def main():
    groundtruth = GroundtruthIterator('/data/sr_results/groundtruth')
    bilinear = BilinearIterator('/data/sr_results/bilinear')

    mse_gan8 = GANIterator(
        outdir = '/data/sr_results/mse/gan-8',
        checkpoint = '/data/sr_models/mse-20210901-111709/training/gan-8'
    )

    rnet_23c_gan3 = GANIterator(
        outdir = '/data/sr_results/rnet-small-23c/gan-3',
        checkpoint = '/data/sr_models/rnet-small-23c-20210912-161623/training/gan-3'
    )

    if True:
        Evaluation(groundtruth, force_overwrite=True).run()
        Evaluation(bilinear, force_overwrite=True).run()
        Evaluation(mse_gan8, force_overwrite=True).run()
        Evaluation(rnet_23c_gan3, force_overwrite=True).run()

    else: 
        Evaluation(groundtruth, force_overwrite=True).summarize('/data/sr_results/summary', {
            'groundtruth': '/data/sr_results/groundtruth',
            #'bilinear': '/data/sr_results/bilinear',
            'mse': '/data/sr_results/mse/gan-8',
            'ours': '/data/sr_results/rnet-small-23c/gan-2'
        })

if __name__ == '__main__':
    main()