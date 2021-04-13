# PHSX815_Project 3: Exoplanet Lightcurves 

This package simulates laightcurve data and attempts to retrieve the parameters of the lightcurve model using an MCMC routine.

## Descriptions of Included Scripts:

* *data_simulator.py*: This script simulates lightcurve data by creating a model with user input, applies some Gaussian noise, and then sotres the data as a csv file. This script takes as input the time of transit (t0), the scaled semi-major axis of the orbit (ar), the scaled planetary radius (rp), the exposure time (exp, in minutes), the error on each data point (error, in terms of normalized flux), and the file name (fname).

* *curve_analysis.py*: This script analyzes data created by *data_simulator.py*. It loads a csv file, optimizes a likelihood object then uses the best fit parameters as an inital position for MCMC walkers. It produces both a best fit plot and corner plot visualizing the posterior distribution in addition to the parameter estimates and the 1 sigma uncertainties.  
## Usage 

To simulate data:

```python
python data_simulator.py -t0 0.0 -ar 30 -rp 0.026 -exp 30 -error 0.0001 -fname test_curve
```

To analyze:

```python
python curve_analysis.py -infile test_curve.csv
``` 


## Dependencies

This code requires:

* Python 3.7.3
* Scipy v1.6.0
* Numpy v1.19.2
* MatplotLib v3.3.2
* TQDM v4.56.0
* batman
* emcee

