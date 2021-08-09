import os

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
        gan = PhIREGANs('eval', mu_sig=[mean, std], print_every=1e9, compression='ZLIB')
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
        gan = PhIREGANs('eval', mu_sig=[mean, std], print_every=1e9, compression='ZLIB')
        
        gan.set_LR_data_shape(dataset)
        H,W,C = gan.LR_data_shape
        LR_in = tf.placeholder(tf.float32, [None, H, W, C])
        bilinear = tf.image.resize(LR_in, [4*H, 4*W], tf.image.ResizeMethod.BILINEAR)

        with tf.Session() as sess:  # will only close if exception occurs inside the coming block

            for idx, LR, HR in gan.iterate_data(dataset, batch_size=2):
                SR = sess.run(bilinear, feed_dict={LR_in: LR})
                yield LR, SR, HR


class GroundtruthIterator:

    def __init__(self, outdir):
        self.outdir = outdir

    def __call__(self, dataset, mean, std):
        gan = PhIREGANs('eval', mu_sig=[mean, std], print_every=1e9, compression='ZLIB')
        for idx, LR, HR in gan.iterate_data(dataset, batch_size=2):
            SR = HR
            yield LR, SR, HR


class Evaluation:

    def __init__(self, iterator):
        self.iterator = iterator
        self.is_groundtruth = isinstance(self.iterator, GroundtruthIterator)

        self.measure_time = False

        self.dataset = sorted(glob('/data/stengel/HR/sr_eval_2000_2002.*.tfrecords'))        
        self.denorm = True

        self.mean_log1p, self.std_log1p = [0.0008630127, 0.0003224114], [0.15800296, 0.16053197]  # alpha=0.2
        #self.mean_log1p, self.std_log1p = [0.008315503, 0.0028762482], [0.5266841, 0.5418187] # alpha=1.0
        self.mean, self.std = [2.0152406e-08, 2.1581373e-07], [2.8560082e-05, 5.0738556e-05]        

        self.setup_metrics()


    def setup_metrics(self):
        self.px_per_deg = 2560 / 360
        img_patch_size = (self.to_px(60), self.to_px(60))
        img_freq = 20

        vis_sea = Visualize((self.to_px(60), self.to_px(90)), img_patch_size, img_freq)
        vis_eu = Visualize((self.to_px(40), self.to_px(0)), img_patch_size, img_freq)
        vis_na = Visualize((self.to_px(40), self.to_px(250)), img_patch_size, img_freq)
        vis_pac = Visualize((self.to_px(60), self.to_px(180)), img_patch_size, img_freq)

        if self.denorm:
            metrics = {
                #'power-spectrum': PowerSpectrum(1280, 2560),
                #'semivariogram': Semivariogram(n_samples=20),
                #'moments': Moments(),
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
        y =  np.sign(y) * np.expm1(np.fabs(y)) / 0.2
        return y*self.std + self.mean


    def create_dirs(self, dir):
        for name, metric in self.metrics.items():
            metric_dir = Path(dir) / name

            try:
                os.makedirs(metric_dir)
            except OSError:
                resp = input(f'{metric_dir} already exists. Continue? (y/n)')
                if resp != 'y' and resp != 'Y':
                    print('Evaluation cancelled')
                    return False
                else:
                    os.makedirs(metric_dir, exist_ok=True)
            
            metric.set_dir(metric_dir)
        
        return True


    def run(self):
        calc_sh = any(metric.needs_sh for metric in self.metrics.values())

        if not self.create_dirs(self.iterator.outdir):
            return

        iter_ = self.iterator(self.dataset, self.mean_log1p, self.std_log1p)

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

            if i==30:
                break

            print(i, flush=True)


        for metric in self.metrics.values():
            metric.finalize()


    def summarize(self, outdir, paths):
        if not self.create_dirs(outdir):
            return

        for metric_name, metric in self.metrics.items():
            metric_paths = {name: Path(path) / metric_name for name,path in paths.items()}
            metric.summarize(metric_paths, metric.dir)



def main():
    groundtruth = GroundtruthIterator('/data/results/srgan/groundtruth-test')
    bilinear = BilinearIterator('/data/results/srgan/bilinear')

    resnet_small_16c_gan17 = GANIterator(
        outdir = '/data/results/srgan/resnet-small-16c/gan17/',
        checkpoint = '/data/results/models/resnet-small-16c-20210622-172045/training/gan-17'
    )

    resnet_small_16c_2xdata_gan17 = GANIterator(
        outdir = '/data/results/srgan/resnet-small-16c-2xdata/gan17/',
        checkpoint = '/data/results/models/resnet-small-16c-2xdata-20210705-055431/training/gan-17'
    )

    resnet_small_16c_2xdata_pre2_gan15 = GANIterator(
        outdir = '/data/results/srgan/resnet-small-16c-2xdata-pre2/gan15',
        checkpoint = '/data/results/models/resnet-small-16c-2xdata-pre2-20210710-134110/training/gan-15'
    )
    
    mse_gan17 = GANIterator(
        outdir = '/data/results/srgan/mse/gan17/',
        checkpoint = '/data/results/models/mse-20210608-094931/training/gan-17'
    )

    if True:
        Evaluation(groundtruth).run()
        #Evaluation(bilinear).run()
        #Evaluation(resnet_small_16c_2xdata_pre2_gan15).run()
    
    else: 
        Evaluation(groundtruth).summarize('summary', {
            'groundtruth': '/data/results/srgan/groundtruth',
            #'resnet-small-16c': '/data/results/srgan/resnet-small-16c/gan17',
            'resnet-small-16c-2xdata': '/data/results/srgan/resnet-small-16c-2xdata/gan17',
            'mse': '/data/results/srgan/mse/gan17'
        })

if __name__ == '__main__':
    main()