import os
import numpy as np
import matplotlib.pyplot as plt
from .base import EvaluationMethod
import imageio
import pyshtools as pysh


def _plot_images(imgs):
    N_MODELS = len(imgs)
    N_IMGS = imgs[list(imgs)[0]].shape[0]

    fig, axes = plt.subplots(N_IMGS, N_MODELS, figsize=(N_MODELS*2.5, N_IMGS*2.5 + 0.1))
    for i, name in enumerate(imgs):
        for j, img in enumerate(imgs[name]):
            ax = axes[j, i] 
            ax.imshow(img)
            ax.axis('off')
            if j == 0:
                ax.set_title(name, fontsize=12) 

    plt.subplots_adjust(wspace=0.02, hspace=0.02)
    return fig, axes


class Visualize(EvaluationMethod):

    def __init__(self, yx=None, patch_size=None, stride=None, highpass=None):
        super(Visualize, self).__init__()

        self.n = 0
        self.yx=yx
        self.patch_size=patch_size
        self.stride = stride or 1
        self.highpass = highpass

        self.needs_sh = True if self.highpass else False
        self.no_groundtruth = False#False if self.highpass else True


    def truncate(self, img, lr=False):
        if self.yx is None or self.patch_size is None:
            return img
        else:
            y,x = self.yx
            h,w = self.patch_size
            if lr:
                y //= 4
                x //= 4
                h //= 4
                w //= 4

            return img[y:y+h, x:x+w]


    def evaluate_SR_sh(self, i, LR, SR, SR_sh):
        if not self.highpass:
            return


        for sh in SR_sh:
            self.n += 1
            if (self.n - 1) % self.stride != 0:
                continue

            # highpass
            sh[:, :self.highpass, ...] = 0      

            div = self.truncate(pysh.expand.MakeGridDH(sh[..., 0], sampling=2))
            vort = self.truncate(pysh.expand.MakeGridDH(sh[..., 1], sampling=2))

            directory = f'{self.dir}/{self.n - 1}'
            os.makedirs(directory, exist_ok=True)

            plt.imsave(f'{directory}/div.png', div, vmin=-1e-5, vmax=1e-5)
            plt.imsave(f'{directory}/vort.png', vort,  vmin=-1e-5, vmax=1e-5)


    def evaluate_both(self, i, LR, SR, HR):
        if self.highpass:
            return

        for img_LR, img_SR, img_HR in zip(LR, SR, HR):
            self.n += 1
            if (self.n - 1) % self.stride != 0:
                continue

            img_LR = self.truncate(img_LR, lr=True)
            img_SR = self.truncate(img_SR)
            img_HR = self.truncate(img_HR)

            directory = f'{self.dir}/{self.n - 1}'
            os.makedirs(directory, exist_ok=True)

            vmin_div, vmax_div = np.min(img_HR[...,0]), np.max(img_HR[...,0])
            vmin_vort, vmax_vort = np.min(img_HR[...,1]), np.max(img_HR[...,1])

            plt.imsave(f'{directory}/div.png', np.hstack([img_SR[...,0], img_HR[...,0]]), vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/vort.png', np.hstack([img_SR[...,1], img_HR[...,1]]), vmin=vmin_vort, vmax=vmax_vort)
            
            plt.imsave(f'{directory}/SR_div.png', img_SR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/SR_vort.png', img_SR[...,1], vmin=vmin_vort, vmax=vmax_vort)

            plt.imsave(f'{directory}/HR_div.png', img_HR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/HR_vort.png', img_HR[...,1], vmin=vmin_vort, vmax=vmax_vort)

            plt.imsave(f'{directory}/LR_div.png', img_LR[...,0], vmin=vmin_div, vmax=vmax_div)
            plt.imsave(f'{directory}/LR_vort.png', img_LR[...,1], vmin=vmin_vort, vmax=vmax_vort)

    
    def summarize(self, paths, outdir):
        N_MAX = 4

        if self.highpass:
            return

        div_imgs = {}
        vort_imgs = {}

        for name, main_path in paths.items():
            if name == 'ground truth':
                continue

            div_imgs[name] = np.stack([imageio.imread(img_p) for img_p in sorted(main_path.glob('*/div.png'))], axis=0)
            vort_imgs[name] = np.stack([imageio.imread(img_p) for img_p in sorted(main_path.glob('*/vort.png'))], axis=0)

            # cutoff HR image. By including the HR image, we ensure that every img has the same cmap though.
            H,W = div_imgs[name].shape[1:3]
            div_imgs['ground truth'] = div_imgs[name][:N_MAX, :, W//2:]
            vort_imgs['ground truth'] = vort_imgs[name][:N_MAX, :, W//2:]

            div_imgs[name] = div_imgs[name][:N_MAX, :, :W//2]
            vort_imgs[name] = vort_imgs[name][:N_MAX, :, :W//2]

        H,W = div_imgs[list(div_imgs)[0]].shape[1:3]

        fig, axes = _plot_images(div_imgs)
        fig.savefig(outdir / 'div.png', bbox_inches='tight', dpi=H//4)
        fig.savefig(outdir / 'div.pdf', bbox_inches='tight')

        fig, axes = _plot_images(vort_imgs)
        fig.savefig(outdir / 'vort.png', bbox_inches='tight', dpi=H//4)
        fig.savefig(outdir / 'vort.pdf', bbox_inches='tight')


        # zoomed in
        #if outdir.parts[-1] == 'img-EU':
        div_imgs = {name: imgs[:, H//2:, W//2:] for name,imgs in div_imgs.items()}
        vort_imgs = {name: imgs[:, H//2:, W//2:] for name,imgs in vort_imgs.items()}

        fig, axes = _plot_images(div_imgs)
        fig.savefig(outdir / 'div_zoom.png', bbox_inches='tight', dpi=H*4)
        fig.savefig(outdir / 'div_zoom.pdf', bbox_inches='tight')

        fig, axes = _plot_images(vort_imgs)
        fig.savefig(outdir / 'vort_zoom.png', bbox_inches='tight', dpi=H*4)
        fig.savefig(outdir / 'vort_zoom.pdf', bbox_inches='tight')