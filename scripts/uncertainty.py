import tensorflow as tf
import numpy as np
import itertools
import math
import os
from glob import glob
from absl import flags
from absl import app
from pathlib import Path

from phire.PhIREGANs import PhIREGANs
from phire.utils import sliding_window_view
from phire.rplearn.skeleton import load_model
from phire.encoder import tf1_custom_objects



FLAGS = flags.FLAGS
flags.DEFINE_integer('offset', 1, 'temporal offset', lower_bound=1)
flags.DEFINE_string('ds', '/data/sr/sr_eval_2000_2005.*.tfrecords', 'glob pattern for the tfrecords files')
flags.DEFINE_string('model_path', '/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27', 'path to pretrained AtmoDist model')
flags.DEFINE_string('outdir', 'uncertainty/', 'path where results are stored')
flags.DEFINE_integer('batch_size', 1024, 'batch size to use for inference')


def batchify(HR1, HR2, batch_size):
    H,W = HR1.shape[:2]
    for start in range(0, H*W, batch_size):
        continuous_idx = np.arange(start, min(start+batch_size, H*W)) 
        i = continuous_idx // W
        j = continuous_idx % W

        hr1_batch = np.transpose(HR1[i,j], (0,2,3,1))
        hr2_batch = np.transpose(HR2[i,j], (0,2,3,1))
        
        yield {'img1_inp': hr1_batch, 'img2_inp': hr2_batch}


def main(argv):
    ds_files = sorted(glob(FLAGS.ds))

    gan = PhIREGANs('eval', None, print_every=1e9, compression='ZLIB')
    ds1 = gan.iterate_data(ds_files, batch_size=1)

    it = iter(ds1)
    queue = []
    for _ in range(FLAGS.offset):
        _,_,X = next(it)
        queue += [X]

    model = load_model(FLAGS.model_path, tf1_custom_objects)

    outdir = Path(FLAGS.outdir)
    os.makedirs(outdir, exist_ok=True)

    predictions = []
    t = 1
    for _,_,HR2 in it:
        HR1 = queue.pop(0)
        queue += [HR2]
        print(f'd={t/8:.1f}')
        
        HR1 = np.pad(HR1[0], ((0,0), (0, 160), (0,0)), 'wrap')
        HR1 = sliding_window_view(HR1, (160,160), (0,1))[::20, ::20]
        
        HR2 = np.pad(HR2[0], ((0,0), (0, 160), (0,0)), 'wrap')
        HR2 = sliding_window_view(HR2, (160,160), (0,1))[::20, ::20]

        H,W = HR1.shape[:2]
        steps = int(math.ceil(H*W / FLAGS.batch_size))
        preds = model.predict(batchify(HR1, HR2, FLAGS.batch_size), steps=steps)
        preds = preds.reshape(H,W,-1)
        
        predictions += [preds]

        if t == 366*8:
            break
        t += 1

    predictions = np.stack(predictions, axis=0)
    np.save(outdir / f'raw_{FLAGS.offset}.npy', predictions, allow_pickle=False)


if __name__ == '__main__':
    app.run(main)
