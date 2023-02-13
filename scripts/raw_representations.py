import tensorflow as tf
import numpy as np

import h5py
import os
from pathlib import Path
from glob import glob

from absl import flags
from absl import app

from phire.PhIREGANs import PhIREGANs
from phire.encoder import load_encoder



FLAGS = flags.FLAGS
flags.DEFINE_string('ds', '/data/sr/sr_eval_2000_2005.*.tfrecords', 'glob pattern for the tfrecords files')
flags.DEFINE_string('model_path', '/data/final_rp_models/rnet-small-23c_2021-09-09_1831/epoch27', 'path to pretrained AtmoDist model')
flags.DEFINE_integer('layer', -1, 'which layer to extract representations from')
flags.DEFINE_string('outfile', 'representations.h5', 'path where results are stored')


def inference(ds, encoder, out_ds):
    t = 0
    for _,_,HR in ds:
        print(f'd={(t+1)/8:.1f}')

        reprs = encoder.predict_on_batch(HR)
        
        if t > 0:
            out_ds.resize(t+1, axis=0)
        out_ds[t,:,:,:] = reprs

        t += 1


def main(argv):
    ds_files = sorted(glob(FLAGS.ds))

    gan = PhIREGANs('eval', None, print_every=1e9, compression='ZLIB')
    in_ds = gan.iterate_data(ds_files, batch_size=1)

    encoder = load_encoder(FLAGS.model_path, FLAGS.layer, (1280,2560,2))

    with h5py.File(FLAGS.outfile, 'w') as f:
        out_ds = f.create_dataset(
            'representations', 
            (1,40,80,128), 
            maxshape=(None,40,80,128), 
            chunks=(4, 40, 80, 128),
            compression="gzip", 
            compression_opts=1
        )
        inference(in_ds, encoder, out_ds)

    #predictions = np.stack(predictions, axis=0)
    #np.save(outdir / f'raw_{FLAGS.offset}.npy', predictions, allow_pickle=False)


if __name__ == '__main__':
    app.run(main)
