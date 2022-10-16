################################
### BOOTSTRAP ANALYSES #########
################################

# from https://towardsdatascience.com/linear-regression-with-bootstrapping-4924c05d2a9
# with some help from: https://stats.stackexchange.com/questions/183230/bootstrapping-confidence-interval-from-a-regression-prediction

'''
This python script contains all the necessary functions used to perform bootstrapping of units and output figures.

Parametric resampling of alpha, beta parameters

INPUTS: 

arr: Numpy array containing unit-specific Vs, effective stress parameters
n_boots: Number of bootstrap sampling iterations to perform
unit: String indicating unit type (Tailings, Compacted & Coarse, Clay) - used solely for image saving.

OUTPUTS: 

boot_beta: List containing appended betas for each Bootstrap (len = n_boots)
boot_alpha: List containing appended alphas for each Bootstrap (len = n_boots)

'''

#Import necessary packages

import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import glob
import pickle
import warnings
from scipy.optimize import curve_fit



def power_law(x, a, b):
    ''' 
    function to calculate the power-law with constants a and b, power regression 
    https://towardsdatascience.com/basic-curve-fitting-of-scientific-data-with-python-9592244a2509
    '''
    return a*np.power(x,b)


def get_ecdf(data):
    '''
    Returns x,y for ecdf
    https://towardsdatascience.com/calculating-confidence-interval-with-bootstrapping-872c657c058d
    '''
    # Get lenght of the data into n
    n = len(data)
    
    # We need to sort the data
    x = np.sort(data)
    
    # the function will show us cumulative percentages of corresponding data points
    y = np.arange(1,n+1)/n
    
    return x,y


def bootstrap(arr, n_boots,unit):
    #nsampls=len(arr[:,-1])                      # if number samples equals full sample size
    nsampls=int(0.7*(len(arr[:,-1])))            # number of samples reduced to 70% - updated August 8 2022

    # POWER LAW REGRESSION ANALYSES
    # Fit the sigma_v' vs Vs power-law data

    pars, cov = curve_fit(f=power_law, xdata=arr[:,-1], ydata=arr[:,2], p0=[0, 0], bounds=(-np.inf, np.inf))
    # Get the standard deviations of the parameters (square roots of the # diagonal of the covariance)
##    stdevs = np.sqrt(np.diag(cov))
    # Calculate the residuals
##    rest = arr[:,2] - power_law(arr[:,-1], *pars)
    alpha=pars[0]
    beta=pars[1]
##    print("Fine tailings regression analyses: ")
##    print("Alpha: %0.2f +/- %0.2f m/s; Beta %0.2f +/- %0.2f" %(alpha_tails,stdevst[0],beta_tails,stdevst[1]))

    # create power-law dataset based on regression parameters alpha, beta
    y_t=np.zeros((len(arr[:,1])))
    temp_t=arr[:,-1]
    x_t = np.sort(temp_t)

    for i in range(len(x_t)):
        y_t[i] = alpha*x_t[i]**beta


    # resample with replacement each row
    boot_beta = []       # BETA
    boot_alpha = []      # ALPHA

    plt.rcParams["figure.figsize"]=(12,6)
    for _ in range(n_boots):
        # sample the rows, same size, with replacement
        sample_index=np.random.choice(range(0,nsampls),nsampls)

        bs_x = arr[:,-1][sample_index]
        bs_y=arr[:,2][sample_index]

        # fit a linear regression
        pars_bs,cov_bs=curve_fit(f=power_law, xdata=bs_x, ydata=bs_y, p0=[0, 0], bounds=(-np.inf, np.inf))

        # append coefficients
        boot_alpha.append(pars_bs[0])
        boot_beta.append(pars_bs[1])

        # for individual units (e.g. compact tailings, tailings, clay)
        y_bst=np.zeros((len(arr[:,1])))
        temp_t=arr[:,-1]
        x_bst = np.sort(temp_t)

        for i in range(len(x_bst)):
            y_bst[i] = boot_alpha[_]*x_bst[i]**boot_beta[_]

        # plot a greyed out line
        plt.plot(x_bst,y_bst,linewidth=2,color='grey',alpha=0.2) # bootstrap simulations

    fsize=18                     #font size for plotting
    plt.plot(x_t,y_t,'--r',linewidth=4,label='%s' %(unit))    # power law regression
    plt.legend()
    plt.plot(arr[:,-1],arr[:,2],'.k')    # Vs sCPT data for fine tailings from 2017/18
    #plt.legend()
    plt.xlabel("Effective vertical stress $\sigma_v'$ (kPa)",fontsize=fsize)
    plt.ylabel("$V_s$ (m/s)",fontsize=fsize)
    plt.tick_params(axis='y', labelsize=fsize) 
    plt.tick_params(axis='x', labelsize=fsize)
    #plt.title('Power regression analyses with bootstrap sampling')
    plt.grid(True)
    plt.savefig('bootstrap_fig_%s.png' % (unit))
    plt.close()
    # need larger bootstrap sample to see tails (rare events) - more robust.

    # plot histogram of alpha, beta obtained from bootstrap sampling
    plt.rcParams["figure.figsize"]=(12,6)
    nbins=50 # number of bins used in histogram

    # plot alpha histogram
    plt.hist(boot_alpha,bins=nbins,color='gray',edgecolor='black')
    # Showing the related percentiles
    plt.axvline(x=np.percentile(boot_alpha,[2.5]), ymin=0, ymax=1,label='2.5th percentile',c='k')
    plt.axvline(x=np.percentile(boot_alpha,[97.5]), ymin=0, ymax=1,label='97.5th percentile',c='k')
    plt.xlabel("Alpha (m/s)",fontsize=fsize)
    plt.ylabel("PDF",fontsize=fsize)
    plt.tick_params(axis='y', labelsize=fsize) 
    plt.tick_params(axis='x', labelsize=fsize)

    #plt.title("Probability Density Function")
    plt.savefig('bootstrap_alpha_hist_%s.png' % (unit))
    plt.close()
    
    # plot beta histogram
    plt.hist(boot_beta,bins=nbins,color='gray',edgecolor='black')
    # Showing the related percentiles
    plt.axvline(x=np.percentile(boot_beta,[2.5]), ymin=0, ymax=1,label='2.5th percentile',c='k')
    plt.axvline(x=np.percentile(boot_beta,[97.5]), ymin=0, ymax=1,label='97.5th percentile',c='k')
    plt.xlabel("Beta",fontsize=fsize)
    plt.ylabel("PDF",fontsize=fsize)
    plt.tick_params(axis='y', labelsize=fsize) 
    plt.tick_params(axis='x', labelsize=fsize)
    #plt.title("Probability Density Function")
    plt.savefig('bootstrap_beta_hist_%s.png' % (unit))
    plt.close()
    
    return boot_alpha, boot_beta

