#!/usr/bin/env python3

#Default Modules:
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import os, glob, sys
from tqdm import tqdm
from astropy import units as u

plt.style.use("/home/custom_style1.mplstyle")

#Other Modules:
from utils import transit_model
from utils import log_likelihood
import emcee
import corner
from scipy.optimize import minimize

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

    labels = ['t0','rp','ar']

    pos = soln.x + 1e-4 * np.random.randn(32, 3)

    nwalkers, ndim = pos.shape

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, args=(time, flux, err))

    sampler.run_mcmc(pos, 1000, progress=True)

    flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

    fig,axes = plt.subplots(3,3,figsize=(8,8))

    corner.corner(flat_samples, labels=labels,fig=fig,labelpad=0.01)


    plt.show()
