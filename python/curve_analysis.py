#!/usr/bin/env python3

#Default Modules:
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import os, glob, sys
from tqdm import tqdm
from astropy import units as u

plt.style.use("./rc_file/custom_style1.mplstyle")

#Other Modules:
from utils import transit_model
from utils import log_likelihood
import emcee
import corner
from scipy.optimize import minimize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
##### Author: Alex Polanski #####


if __name__ == "__main__":


    # Input stuff

    if '-infile' in sys.argv:
        p = sys.argv.index('-infile')
        data_file = str(sys.argv[p+1])


    data = pd.read_csv(f'./curves/{data_file}')

    time, flux, err = data['time'].to_numpy(), data['flux'].to_numpy(), data['flux_error'].to_numpy()
    
    # Perform a likelihood fit

    guesses = [0.001,0.019,30]

    nll = lambda *args: -log_likelihood(*args)

    soln = minimize(nll, guesses,method="Powell", args = ( time, flux, err))
    
    print(soln)

    # MCMC

    labels = ['$t_0$','$R_P/R_*$','$a/R_*$']

    pos = soln.x + 1e-4 * np.random.randn(32, 3)

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(time, flux, err))

    sampler.run_mcmc(pos, 1000, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    # Get the Median value of the posteriors and the uncertainties 

    medians = []
    quants = []
    per_error = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        medians.append(mcmc[1])
        quants.append([q[0],q[1]])
        per_error.append((np.mean([q[0],q[1]])/mcmc[1])*100.0)

    print(f'\nMedian Parameters, Uncertainties, Percent Error:\n t0: {medians[0]:0.3} +{quants[0][0]:0.3} -{quants[0][1]:0.3} | {np.abs(per_error[0]):0.2}\n Rp/R*: {medians[1]:0.3} +{quants[1][0]:0.3} -{quants[1][1]:0.3} | {np.abs(per_error[1]):0.2}\n a/R*: {medians[2]:0.3} +{quants[2][0]:0.3} -{quants[2][1]:0.3} | {np.abs(per_error[2]):0.2}\n')




    fig_corner,axes = plt.subplots(3,3,figsize=(10,10))

    corner.corner(flat_samples, labels=labels,fig=fig_corner,labelpad=0.1, truths=medians)

    for i in range(ndim):
        axe = axes[i,i]
        axe.axvline(medians[i])
        
        axe.axvline(medians[i]-quants[i][0],linestyle='--')
        axe.axvline(medians[i]+quants[i][1],linestyle='--')


    fig, ax = plt.subplots(figsize=(15,10))
    t = np.linspace(np.min(time),np.max(time), 1000)
    ax.errorbar(time, flux, yerr=err, fmt='.k', label='Data')
    ax.plot(t, transit_model(t, medians[0],medians[1],medians[2]), linewidth = 3.0, label='Median Model')
    ax.set_xlabel("Time from Transit Center (days)")
    ax.set_ylabel("Normalized Flux")
    ax.legend(loc='upper right')

    #inset residual plot

    axin = inset_axes(ax, width="25%", height="15%", loc='lower right')
    axin.hist((flux-transit_model(time, medians[0],medians[1],medians[2])))
    axin.tick_params(labelleft=False, labelbottom=False)
    axin.axvline(0.0, linestyle='--',color='black')
    axin.set_title("Residuals", loc='left')




    plt.show()
