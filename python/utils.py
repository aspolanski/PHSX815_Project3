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
import batman

##### Author: Alex Polanski #####

def log_likelihood(theta, x, y, yerr):

    t0, rp, ar = theta

    model_y = transit_model(x, t0, rp, ar)

    return( -0.5 * np.sum((model_y - y)**2/ yerr**2 + np.log(2*np.pi*yerr**2)) )
    #return( -0.5 * np.sum(np.square(model_y - y)/yerr**2 + np.log(2*np.pi*yerr**2)) )

def transit_model(time, t0, rp, ar):

    params = batman.TransitParams()
    params.t0 = t0                       #time of inferior conjunction
    params.per = 15                     #orbital period
    params.rp = rp                      #planet radius (in units of stellar radii)
    params.a = ar                       #semi-major axis (in units of stellar radii)
    params.inc = 89                     #orbital inclination (in degrees)
    params.ecc = 0                      #eccentricity
    params.w = 0                       #longitude of periastron (in degrees)
    params.u = [0.3288,0.3024]                #limb darkening coefficients [u1, u2]
    params.limb_dark = "quadratic"       #limb darkening model

    m = batman.TransitModel(params,time,transittype='primary')
    flux = m.light_curve(params)

    return flux




