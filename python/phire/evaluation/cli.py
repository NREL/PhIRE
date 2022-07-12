import matplotlib
matplotlib.use('Agg')  # this results in considerable plotting speedup and enables multiprocessing + non-interactive work

import os
import sys
import gc

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import numpy as np
import pandas as pd

from glob import glob
from pathlib import Path
from time import time
import pyshtools as pysh
from scipy.ndimage import gaussian_filter1d
import scipy.signal
import pkg_resources
import matplotlib.pyplot as plt
import multiprocessing
import traceback

from phire.PhIREGANs import PhIREGANs
import phire.utils as utils

from .visualize import Visualize
from .moments import Moments
from .semivariogram import Semivariogram
from .spectrum import PowerSpectrum
from .histogram import Histogram
from .highpass import HighpassCounter
from .project import Project
from .temporal_metric import TemporalMetric


BATCH_SIZE = 1

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
            batch_size=BATCH_SIZE, 
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

        for idx, LR, HR in gan.iterate_data(dataset, batch_size=BATCH_SIZE):
            SR = tf.get_default_session().run(bilinear, feed_dict={LR_in: LR})
            yield LR, SR, HR


class GroundtruthIterator:

    def __init__(self, outdir):
        self.outdir = outdir

    def __call__(self, dataset, mean, std):
        mu_sig = [mean, std] if mean and std else None
        gan = PhIREGANs('eval', mu_sig, print_every=1e9, compression='ZLIB')
        for idx, LR, HR in gan.iterate_data(dataset, batch_size=BATCH_SIZE):
            SR = HR
            yield LR, SR, HR


def _process_metric(paths, metric_name, metric):
    """
    Helper function used by Evaluation.summarize().
    Needs to be global to be pickleable.
    """
    print(paths, metric_name, metric)
    try:
        print(f'processing {metric_name} ...', flush=True)
        metric_paths = {name: Path(path) / metric_name for name,path in paths.items() if (Path(path) / metric_name).exists()}
        metric.summarize(metric_paths, metric.dir)
    except:
        traceback.print_exc()
        sys.stdout.flush()


# have to be global to be pickleable
def _mse_loss(SR,HR): 
    return np.mean((SR-HR)**2, axis=(1,2))

def _tv_loss(SR,HR): 
    return utils.tv(SR) - utils.tv(HR)

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


    def setup_metrics(self):
        self.px_per_deg = 2560 / 360
        img_patch_size = (self.to_px(60), self.to_px(60))
        img_freq = 1000

        vis_sea =   Visualize((self.to_px(60), self.to_px(90)), img_patch_size, img_freq)
        vis_eu =    Visualize((self.to_px(40), self.to_px(0)), img_patch_size, img_freq)
        vis_na =    Visualize((self.to_px(40), self.to_px(250)), img_patch_size, img_freq)
        vis_pac =   Visualize((self.to_px(60), self.to_px(180)), img_patch_size, img_freq)

        """
        P = np.load(pkg_resources.resource_filename('phire', 'data/gaussian_projection.npy'))
        rand_proj = Project(P, self.mean, self.std)

        P = np.load(pkg_resources.resource_filename('phire', 'data/pca_projection.npy'))
        pca_proj = Project(P, self.mean, self.std)
        """

        if self.denorm:
            self.metrics = {
                #'power-spectrum': PowerSpectrum(1280, 2560),
                #'semivariogram': Semivariogram(n_samples=300),
                # 'moments': Moments(),
                #'imgs/SEA': vis_sea,
                #'imgs/EU': vis_eu,
                #'imgs/NA': vis_na,
                #'imgs/pacific': vis_pac,
                'losses/mse': TemporalMetric(_mse_loss, label='mean squared error'),
                'losses/tv': TemporalMetric(_tv_loss, label='total variation difference'),
                #'random_projection': rand_proj,
                #'pca_projection': pca_proj
            }

            cities = pd.read_csv(pkg_resources.resource_filename('phire', 'data/cities.csv'))
            for i, city in cities.iterrows():
                y = self.to_px(90 - city.lat)
                x = self.to_px(city.lng) if city.lng >= 0 else self.to_px(180 - city.lng)
                machine_name = city.city_ascii.replace(' ', '_').lower()

                if i < 50:
                    projection = lambda batch, y=y,x=x: batch[:, y:y+25, x:x+25, :]
                    #self.metrics[f'projections/{machine_name}'] = Project(None, self.mean, self.std, projection)
                
                #self.metrics[f'histograms_1x1/{machine_name}'] = Histogram((y,x), (1,1))
                #self.metrics[f'histograms_2x2/{machine_name}'] = Histogram((y,x), (2,2))
                #self.metrics[f'histograms_3x3/{machine_name}'] = Histogram((y,x), (3,3))


        else:
            self.metrics = {
                'imgs-transform/SEA': vis_sea,
                'imgs-transform/EU': vis_eu,
                'imgs-transform/NA': vis_na,
                'imgs-transform/pacific': vis_pac,
            }        


    def to_px(self, deg):
        return int(round(deg*self.px_per_deg))


    def deprocess(self, batch):
        y = batch * self.std_log1p + self.mean_log1p
        y =  np.sign(y) * np.expm1(np.fabs(y)) / 0.2  # alpha=0.2
        y = y*self.std + self.mean

        return y
        

    def create_dirs(self, dir, metrics):
        for name, metric in metrics.items():
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



    def run(self, max_iters=None):
        # filter metrics:
        metrics = {k:metric for k, metric in self.metrics.items() if not metric.no_groundtruth or not self.is_groundtruth}
        calc_sh = any(metric.needs_sh for metric in metrics.values())

        self.test_dims()

        if not self.create_dirs(self.iterator.outdir, metrics):
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
                    HR_sh = [np.stack([pysh.expand.SHExpandDH(img) for img in HR[..., c]], axis=0) for c in range(C)]
                    HR_sh = np.stack(HR_sh, axis=-1)
                    t2 = time()
                    
                    if self.measure_time:
                        print(f'sh-transform took {t2-t1:.2f}s')


                for name, metric in metrics.items():
                    if i == 0:
                        metric.set_shape(HR.shape[1:])

                    t1 = time()
                    if metric.needs_sh:
                        metric.evaluate_both_sh(i, LR, SR, HR, SR_sh, HR_sh)
                        metric.evaluate_SR_sh(i, LR, SR, SR_sh)
                    else:
                        metric.evaluate_both(i, LR, SR, HR)
                        metric.evaluate_SR(i, LR, SR)
                    t2 = time()

                    if self.measure_time:
                        print(f'{name} took {t2-t1:.2f}s')

                print(f'\r{i}', flush=True, end='')
                if max_iters and i+1 == max_iters:
                    break


            for metric in metrics.values():
                metric.finalize()

        finally:
            # make sure that any ressources (tf.Session) held by iter_ get released now!
            # this is important
            del iter_


    def summarize(self, outdir, paths):
        if not self.create_dirs(outdir, self.metrics):
            return

        with multiprocessing.Pool(8) as pool:
            results = []
            for name, metric in self.metrics.items():
                args =  [paths, name, metric]
                results.append(pool.apply_async(_process_metric, args))
            
            # wait for everything to finish
            for res in results:
                res.wait()  # don't call get() here

            # ensure that exceptions thrown during serialization/deserailization get propagated
            for res in results:
                res.get()


def main():
    plt.ioff()

    DIR = Path('/data/sr_results')

    groundtruth = GroundtruthIterator(DIR / 'groundtruth')
    bilinear = BilinearIterator(DIR / 'bilinear')

    mse_gan18 = GANIterator(
        outdir = DIR / 'mse/gan-18',
        checkpoint = '/data/sr_models/mse-20210901-111709/training/gan-18'
    )

    rnet_23c_gan18 = GANIterator(
        outdir = DIR / 'rnet-small-23c/gan-18',
        checkpoint = '/data/sr_models/rnet-small-23c-20210912-161623/training/gan-18'
    )

    abla_15c_gan9 = GANIterator(
        outdir = DIR / 'abla-15c/gan-9',
        checkpoint = '/data/sr_models/abla-15c-20211010-150809/training/gan-9'
    )

    abla_23c_gan9 = GANIterator(
        outdir = DIR / 'abla-23c/gan-9',
        checkpoint = '/data/sr_models/abla-23c-20211010-150650/training/gan-9'
    )

    abla_31c_gan9 = GANIterator(
        outdir = DIR / 'abla-31c/gan-9',
        checkpoint = '/data/sr_models/abla-31c-20211011-134752/training/gan-9'
    )

    fftreg = GANIterator(
        outdir = DIR / 'rnet-small-23c-fftreg/gan-6',
        checkpoint = '/data/sr_models/rnet-small-23c-fftreg-20210924-142214/training/gan-6'
    )

    if False:
        """
        Evaluation(groundtruth, force_overwrite=True).run(max_iters=None)
        del groundtruth
        gc.collect()
        """
        

        """
        #Evaluation(bilinear, force_overwrite=True).run(max_iters=4)
        
        Evaluation(mse_gan18, force_overwrite=True).run(max_iters=None)
        del mse_gan18
        gc.collect()
        
        Evaluation(rnet_23c_gan18, force_overwrite=True).run(max_iters=None)
        del rnet_23c_gan18
        gc.collect()

        Evaluation(abla_15c_gan9, force_overwrite=True).run(max_iters=None)
        del abla_15c_gan9
        gc.collect()

        Evaluation(abla_23c_gan9, force_overwrite=True).run(max_iters=None)
        del abla_23c_gan9
        gc.collect()

        Evaluation(abla_31c_gan9, force_overwrite=True).run(max_iters=None)
        del abla_31c_gan9
        gc.collect()
        """
        
        #Evaluation(fftreg, force_overwrite=True).run(max_iters=30)
        pass

    if True: 
        Evaluation(groundtruth, force_overwrite=True).summarize(DIR / 'summary2', {
            'ground truth': DIR / 'groundtruth',
            'ours': DIR / 'rnet-small-23c/gan-18',
            'l2': DIR / 'mse/gan-18',
        })

        """
        Evaluation(groundtruth, force_overwrite=True).summarize(DIR / 'ablation-summary2', {
            'ground truth': DIR / 'groundtruth',
            '45h': DIR / 'abla-15c/gan-9',
            '69h': DIR / 'abla-23c/gan-9',
            '93h': DIR / 'abla-31c/gan-9',
            'mse': DIR/ 'mse/gan-9'
        })
        """

if __name__ == '__main__':
    main()