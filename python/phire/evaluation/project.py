import numpy as np
from .base import EvaluationMethod
import ot
import json
import scipy


def wasserstein(X,Y,metric):
    M = ot.dist(X, Y, metric=metric)
    M /= np.max(M)

    n1,n2 = M.shape
    a = np.ones(n1) / n1  # 1d histogram, uniform distribution
    b = np.ones(n2) / n2 

    return ot.emd2(a,b,M)


def frechet(u1,u2, C1, C2):
    Lambda, U = np.linalg.eig(C1@C2)
    Lambda[Lambda < 0] = 0  # C1 and C2 are positive semi-definit
    Lambda = np.sqrt(Lambda)
    sqrtm = np.real(U @ np.diag(Lambda) @ np.linalg.inv(U))  # matrix square-root of C1*C2

    return np.dot(u1-u2, u1-u2) + np.trace(C1+C2-2*sqrtm)


def rv_coefficient(X,Y):
    all_cov = np.cov(X,Y,rowvar=False)
    
    K1 = X.shape[1]
    cov_XX = all_cov[:K1, :K1]
    cov_YY = all_cov[K1:, K1:]
    cov_XY = all_cov[:K1, K1:]
    cov_YX = all_cov[K1:, :K1]
    
    COVV = np.trace(cov_XY @ cov_YX)
    VAV_X = np.trace(cov_XX @ cov_XX)
    VAV_Y = np.trace(cov_YY @ cov_YY)
    return COVV / np.sqrt(VAV_X*VAV_Y)


class Project(EvaluationMethod):

    def __init__(self, proj_matrix, mean, std, transform=None):
        """
        proj_matrix: Projection matrix of shape [H*W*C, K]
        mean: mean vector of data
        std: standard deviation of data
        """

        super(Project, self).__init__()
        self.proj_matrix = proj_matrix
        self.mean = mean
        self.std = std
        self.transform = transform

        self.projections = []

    
    def evaluate_SR(self, i, LR, SR):
        N,H,W,C = SR.shape
        normed = (SR - self.mean) / self.std

        if self.transform:
            projected = self.transform(normed).reshape(N, -1)
        else:
            projected = normed.reshape(N, -1) @ self.proj_matrix  # N x K
        self.projections.append(projected)


    def finalize(self):
        projs = np.concatenate(self.projections, axis=0)
        np.save(self.dir / 'projected.npy', projs, allow_pickle=False)


    def summarize(self, paths, outdir):
        if 'ground truth' not in paths:
            return

        wdist_l2 = {}
        wdist_cosine = {}
        fdist = {}
        rv_coeffs = {}

        groundtruth = np.load(paths['ground truth'] / 'projected.npy')
        C1 = np.cov(groundtruth, rowvar=False)
        u1 = np.mean(groundtruth, axis=0)
        for name in paths:
            samples = np.load(paths[name] / 'projected.npy')

            # Wasserstein distance
            wdist_l2[name] = wasserstein(groundtruth, samples, 'euclidean')
            wdist_cosine[name] = wasserstein(groundtruth, samples, 'cosine')

            # Fréchet distance 
            C2 = np.cov(samples, rowvar=False)
            u2 = np.mean(samples, axis=0)
            fdist[name] = frechet(u1, u2, C1, C2)
            np.savetxt(outdir / f'cov_{name}.csv', C2)

            # RV-Coefficient
            rv_coeffs[name] = rv_coefficient(groundtruth, samples)

        with open(outdir / 'wasserstein_distances_l2.json', 'w') as f:
            json.dump(wdist_l2, f, indent=4)

        with open(outdir / 'wasserstein_distances_cosine.json', 'w') as f:
            json.dump(wdist_cosine, f, indent=4)

        with open(outdir / 'frechet_distances.json', 'w') as f:
            json.dump(fdist, f, indent=4)

        with open(outdir / 'rv_coefficients.json', 'w') as f:
            json.dump(rv_coeffs, f, indent=4)