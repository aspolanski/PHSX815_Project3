#!/usr/bin/env python3

#Default Modules:
import numpy as np
import pandas as pd
import scipy 
import matplotlib.pyplot as plt
import os, glob, sys
from tqdm import tqdm
from astropy import units as u

#Other Modules:
from utils import transit_model

##### Author: Alex Polanski #####


if __name__ == "__main__":

    # Input stuff

    if '-t0' in sys.argv:
        p = sys.argv.index('-t0')
        t0 = float(sys.argv[p+1])

    if '-ar' in sys.argv:
        p = sys.argv.index('-ar')
        ar = float(sys.argv[p+1])
	
    if '-rp' in sys.argv:
        p = sys.argv.index('-rp')
        rp = float(sys.argv[p+1])

    if '-rate' in sys.argv:
        p = sys.argv.index('-rate')
        rate = int(sys.argv[p+1])

    if '-error' in sys.argv:
        p = sys.argv.index('-error')
        error = float(sys.argv[p+1])
    
    if '-fname' in sys.argv:
        p = sys.argv.index('-fname')
        fname = str(sys.argv[p+1])
    

    time = np.linspace(-0.25,0.25,rate)

    flux = transit_model(time,t0,rp,ar) + np.random.normal(0.0,1e-5,len(time))
    flux_err = np.ones(len(time)) * error

    data_frame = pd.DataFrame(data={'time':time, 'flux':flux, 'flux_error':flux_err})

    data_path = './curves/' + fname + '.csv'

    data_frame.to_csv(data_path, index=False)

