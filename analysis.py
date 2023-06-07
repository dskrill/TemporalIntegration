import torch
import numpy as np
from scipy.optimize import curve_fit
from tqdm.auto import tqdm

def fit_func(x, a, b, c):
    """The convex combination of exponential and power-law functions."""
    return c * (x + 1)**(-a) + (1 - c) * np.exp(-b * x)

def fit_curves(D,bidir=False,pretrained=True):
    """Extracts and fits integration windows as a convex combination of exponential and power law.

    Args:
        D (torch.Tensor): tensor of activations with shape (n_features, n_stim_time, n_model_time, n_layers)
        bidir (bool): whether to use bidirectional windows
        pretrained (bool): Was D obtained using a pretrained model? Random models yield integration windows
                            that are practically delta functions, so modeling them as exp./convex does not make sense.

    Returns:
        all_fits (list): list of fit parameters for each layer
        all_D_delta (list): list of integration windows for each layer
    """


    all_fits = []
    all_D_delta = []
    n_features, n_stim_time, n_model_time, n_layers = D.shape
    for j in tqdm(range(n_features)):
        D_ = D[j,:,:,:].copy()
        for i in range(n_layers):
            D_[:,:, i] = D_[:, :, i] / np.median(np.diag(D_[:, :, i]))

        # model vs. stimulus time matrix
        max_window_size = 21
        n_model_timepoints = D_.shape[1]
        D_delta = np.full((max_window_size, n_model_time, n_layers), np.nan)
        for i in range(n_layers):
            for j in range(n_model_timepoints):
                xi = j - np.arange(max_window_size)
                if all(xi > 1):
                    D_delta[:, j, i] = D_[xi, j, i]
                if bidir:
                    xj = j + np.arange(max_window_size)
                    if all(xj < n_model_timepoints):
                        D_delta[:, j, i] = D_[xj, j, i]
        xi = ~np.isnan(D_delta[0, :, 0])
        D_delta = D_delta[:, xi, :]

        # Rescale by value at t=0
        D_delta /= D_delta[0:1,:,:]


        if pretrained and not bidir:
            # fit as convex combination of power and exponential
            fitobj = []
            for i in range(n_layers):
                t = np.arange(max_window_size)
                y = np.median(D_delta[:, :, i], axis=1)
                bounds = ([0,0, 0], [10, 10, 1])
                p0 = [1, 1, 0.5]
                popt, _ = curve_fit(fit_func, t, y, p0=p0, bounds=bounds)
                fitobj.append(popt)
            all_fits.append(fitobj)
        all_D_delta.append(D_delta)
    return all_fits, all_D_delta
