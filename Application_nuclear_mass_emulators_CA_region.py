
# coding: utf-8

# In[1]:


# Python modules required to run the code for example: Averaging nuclear mass emulators in the Ca region
import numpy as np
import pandas as pd
import pymc3 as pm 
import matplotlib.pyplot as plt
import theano
import sys
import scipy as sp
#import theano.tensor as tt
from tqdm import tqdm  # Loop counter
import os
#import re
import glob
from theano.ifelse import ifelse
import pickle


# In[7]:


def credible_interval(data, alpha, ci_type = 'hpd'):
    """Function calculates bayesian credible intervals of various kinds
    
    Args:
        data: Numpy array of posterior samples (each row corrsponds to one nucleus)
        alpha: Level
        ci_type: 'hpd' for highest posterior density interval, 'et' for equal-tail interval, 
                 'm' for interval centered at mean.
                 
    Returns:
        ci: Numpy array with CI in each row
    """
    if ci_type == 'hpd':
        ci = pm.stats.hpd(data, alpha = alpha)
    if ci_type == 'et':
        ci = np.array(sp.stats.mstats.mquantiles(data, axis= 0,  prob = [alpha/2, 1- alpha/2])).T
    if ci_type == 'm':
        ci = np.array(sp.stats.mstats.mquantiles(data - np.array(sp.stats.mstats.mquantiles(data, axis= 0,  prob = [0.5])) + np.mean(data, axis = 0),
                                                 axis= 0,  prob = [alpha/2, 1- alpha/2])).T
    return ci 


# This part is where we calculate evidence integrals:
def evidence_integral(even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N ,model = "UNEDF1" ,
                      domain = "ca", n_mc = 10000):
    """Function calculate evidence integral for given model. The results are saved as .npy file.
    
    Args:
        even_Z_odd_N: Data with even-Z odd-N nuclei.
        odd_Z_odd_N: Data with odd-Z odd-N nuclei.
        even_Z_even_N: Data with even-Z even-N nuclei.
        odd_Z_even_N: Data with odd-Z even-N nuclei.
        model: One of 'SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24'
        domain: "ca" for the calcuim region
        n_mc: Number of MC samples  to approximate the evidence integrals
        
    Return:
        True
    """
    
    even_Z_odd_N = even_Z_odd_N[["train2003", model, "Z", "N"]]
    odd_Z_odd_N = odd_Z_odd_N[["train2003", model, "Z", "N"]]
    even_Z_even_N = even_Z_even_N[["train2003", model, "Z", "N"]]
    odd_Z_even_N = odd_Z_even_N[["train2003", model, "Z", "N"]]

    even_Z_odd_N_data = even_Z_odd_N.values.astype(float)
    odd_Z_odd_N_data = odd_Z_odd_N.values.astype(float)
    even_Z_even_N_data = even_Z_even_N.values.astype(float)
    odd_Z_even_N_data = odd_Z_even_N.values.astype(float)


    if domain == "ca":
        even_Z_odd_N_data = even_Z_odd_N_data[(even_Z_odd_N_data[:,2] <= 22) & (even_Z_odd_N_data[:,2] >= 14)]
        odd_Z_odd_N_data = odd_Z_odd_N_data[(odd_Z_odd_N_data[:,2] <= 22) & (odd_Z_odd_N_data[:,2] >= 14)]
        even_Z_even_N_data = even_Z_even_N_data[(even_Z_even_N_data[:,2] <= 22) & (even_Z_even_N_data[:,2] >= 14)]
        odd_Z_even_N_data = odd_Z_even_N_data[(odd_Z_even_N_data[:,2] <= 22) & (odd_Z_even_N_data[:,2] >= 14)]
        
    for observable in ["eo", "oo", "ee", "oe"]:
        
        if observable == "eo":
            y = even_Z_odd_N_data[:,0]
            X = even_Z_odd_N_data[:,1:]
        elif observable == "oo":
            y = odd_Z_odd_N_data[:,0]
            X = odd_Z_odd_N_data[:,1:]
        elif observable == "ee":
            y = even_Z_even_N_data[:,0]
            X = even_Z_even_N_data[:,1:]
        elif observable == "oe":
            y = odd_Z_even_N_data[:,0]
            X = odd_Z_even_N_data[:,1:]

        with pm.Model() as gp_mass:

            # Priors
            eta = pm.Gamma("eta", alpha = 0.8, beta = 1, transform=None)
            rho = pm.Gamma("rho", alpha = [0.5, 1.8], beta = [1, 1], shape = 2, transform=None)

            # gp defintion
            coeffs = [1, 0, 0]
            gp_mean = pm.gp.mean.Linear(coeffs = coeffs, intercept = 0)
            gp_cov = eta ** 2 * pm.gp.cov.ExpQuad(X.shape[1], ls = rho, active_dims = [1, 2])
            gp_model = pm.gp.Marginal(mean_func = gp_mean, cov_func = gp_cov)

            y_gp = gp_model.marginal_likelihood("y", X = X, y = y, noise = 0.01, transform=None)
        
        np.random.seed(0)

        if True:

            with pm.Model() as priors_model:

                # Priors
                eta = pm.Gamma("eta", alpha = 0.8, beta = 1, transform=None)
                rho = pm.Gamma("rho", alpha = [0.5, 1.8], beta = [1, 1], shape = 2, transform=None)
                trace_priors = pm.sample(n_mc, tune = 10000, chains = 1)


            log_likelihood = np.empty(0)
            mc_integral = np.empty(n_mc)

            logp = y_gp.logp

            for i in tqdm(range(n_mc), desc = "Log likelihood eval"):
                log_likelihood = np.append(log_likelihood, logp(trace_priors[i], transfrom = None))

            for i in tqdm(range(n_mc), desc = "Integral calc"):
                m = max(log_likelihood[:(i + 1)])
                mc_integral[i] = (np.exp(m) * np.sum(np.exp(log_likelihood[:(i + 1)] - m))) / (i + 1)
                
            name = model + "_" + domain + "_" + str(n_mc) + "_1_" + "_observable_" + observable
            np.save("evidence_" + name, mc_integral)
    return True

def posterior_train(posteriors_dict, even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N ,
                    n_mc = 200000, domain = "ca"):
    """Samples the BMA posterior for the training dataset based on the calculated evidence for the integrals.
    This function is used for AME2003 data
    
    Args:
        posteriors_dict: Dictionary of posterior samples for each of the model based on Neufcourt et. al. (2019)
        even_Z_odd_N: Data with even-Z odd-N nuclei.
        odd_Z_odd_N: Data with odd-Z odd-N nuclei.
        even_Z_even_N: Data with even-Z even-N nuclei.
        odd_Z_even_N: Data with odd-Z even-N nuclei.
        model: One of 'SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24'
        domain: "ca" for the calcuim region
        n_mc: Number of MC samples  to approximate the evidence integrals
        
    Return:
        True
    """
    
    if domain == "ca":
        even_Z_odd_N = even_Z_odd_N[(even_Z_odd_N.Z <= 22) & (even_Z_odd_N.Z >= 14)]
        odd_Z_odd_N = odd_Z_odd_N[(odd_Z_odd_N.Z <= 22) & (odd_Z_odd_N.Z >= 14)]
        even_Z_even_N = even_Z_even_N[(even_Z_even_N.Z <= 22) & (even_Z_even_N.Z >= 14)]
        odd_Z_even_N = odd_Z_even_N[(odd_Z_even_N.Z <= 22) & (odd_Z_even_N.Z >= 14)]
        # Calculate the posteriors on model
    
    
    model_list = ['SKM', 'SKP','SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24']
    alpha = 1
    observable = "eo"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("even-Z, odd-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('even_Z_odd_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_train_20", even_Z_odd_N_posterior)

    alpha = 1
    observable = "oo"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("odd-Z, odd-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('odd_Z_odd_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_train_20", odd_Z_odd_N_posterior)

    alpha = 1
    observable = "ee"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("even-Z, even-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('even_Z_even_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_train_20", even_Z_even_N_posterior)

    alpha = 1
    observable = "oe"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("odd-Z, even-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break
    np.save('odd_Z_even_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_train_20", odd_Z_even_N_posterior)
    
    return True

def posterior_test(posteriors_dict, even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N ,
                    n_mc = 200000, domain = "ca"):
    """Samples the BMA posterior for the training dataset based on the calculated evidence for the integrals.
    This function is used for AME2016 data
    
    Args:
        posteriors_dict: Dictionary of posterior samples for each of the model based on Neufcourt et. al. (2019)
        even_Z_odd_N: Data with even-Z odd-N nuclei.
        odd_Z_odd_N: Data with odd-Z odd-N nuclei.
        even_Z_even_N: Data with even-Z even-N nuclei.
        odd_Z_even_N: Data with odd-Z even-N nuclei.
        model: One of 'SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24'
        domain: "ca" for the calcuim region
        n_mc: Number of MC samples  to approximate the evidence integrals
        
    Return:
        True
    """
    # Calculate the posteriors on model
    model_list = ['SKM', 'SKP','SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24']
    if domain == "ca":
        even_Z_odd_N = even_Z_odd_N[(even_Z_odd_N.Z <= 22) & (even_Z_odd_N.Z >= 14)]
        odd_Z_odd_N = odd_Z_odd_N[(odd_Z_odd_N.Z <= 22) & (odd_Z_odd_N.Z >= 14)]
        even_Z_even_N = even_Z_even_N[(even_Z_even_N.Z <= 22) & (even_Z_even_N.Z >= 14)]
        odd_Z_even_N = odd_Z_even_N[(odd_Z_even_N.Z <= 22) & (odd_Z_even_N.Z >= 14)]
        # Calculate the posteriors on model

    alpha = 1
    observable = "eo"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("even-Z, odd-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('even_Z_odd_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_test_20", even_Z_odd_N_posterior)

    alpha = 1
    observable = "oo"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("odd-Z, odd-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('odd_Z_odd_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_test_20", odd_Z_odd_N_posterior)

    alpha = 1
    observable = "ee"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("even-Z, even-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break

    np.save('even_Z_even_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_test_20", even_Z_even_N_posterior)


    alpha = 1
    observable = "oe"
    model_evidence_dict = {}
    # Getting the evidence integrals
    for model_name in model_list:
        evidence = np.load("evidence_" + model_name + "_" + domain + "_" + str(n_mc) + "_" + str(alpha) + "_observable_" + observable + ".npy")
        model_evidence_dict[model_name] = evidence[-1]

    # Dictionary comprehension to calculate the posteriors
    # Note that the priors for models are uniform 
    model_posteriors_dict = {k: v / sum(model_evidence_dict.values(), 0.0) for k,v in model_evidence_dict.items()}
    print("odd-Z, even-N: ")
    print(model_posteriors_dict)

    # innitialize posterior samples
    np.random.seed(0)
    even_Z_odd_N_posterior = np.empty((even_Z_odd_N.shape[0],0))
    odd_Z_odd_N_posterior = np.empty((odd_Z_odd_N.shape[0],0))
    even_Z_even_N_posterior = np.empty((even_Z_even_N.shape[0],0))
    odd_Z_even_N_posterior = np.empty((odd_Z_even_N.shape[0],0))

    # initialize iterator through dictionaries
    # this represents the next sample to be gathered
    posterior_indicators = {k: 0 for k,v in model_posteriors_dict.items()}

    for i in range(20000):
        upper_boud = 0
        u = np.random.uniform()
        # This is how I decide which posterior to take
        for k, v in model_posteriors_dict.items():
            upper_boud = upper_boud + v
            if u < upper_boud:
                posterior_indicators[k] = i
                even_Z_odd_N_new = posteriors_dict['even_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_odd_N_posterior = np.concatenate([even_Z_odd_N_posterior, even_Z_odd_N_new], axis = 1)

                odd_Z_odd_N_new = posteriors_dict['odd_Z_odd_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_odd_N_posterior = np.concatenate([odd_Z_odd_N_posterior, odd_Z_odd_N_new], axis = 1)

                even_Z_even_N_new = posteriors_dict['even_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                even_Z_even_N_posterior = np.concatenate([even_Z_even_N_posterior, even_Z_even_N_new], axis = 1)

                odd_Z_even_N_new = posteriors_dict['odd_Z_even_N_' + k].iloc[:, posterior_indicators[k]][:, None]
                odd_Z_even_N_posterior = np.concatenate([odd_Z_even_N_posterior, odd_Z_even_N_new], axis = 1)

                break
    np.save('odd_Z_even_N_new_' + '_observable_' + observable + "_alpha_" + str(alpha) + "_test_20", odd_Z_even_N_posterior)


# In[36]:


# Preprocessing of AME2003 of one-neutron and two-neutron separation energies to calculate the evidence integral
even_Z_odd_N = pd.read_csv("data_S1n_2018_even_Z_odd_N.csv")
even_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_odd_N = pd.read_csv("data_S1n_2018_odd_Z_odd_N.csv")
odd_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_even_N = pd.read_csv("data_S2n_2018_even_Z_even_N.csv")
even_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_even_N = pd.read_csv("data_S2n_2018_odd_Z_even_N.csv")
odd_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)

# Filtration for the values 
even_Z_odd_N = even_Z_odd_N.assign(ids = pd.Series(np.array([1] * len(even_Z_odd_N))))
even_Z_odd_N = even_Z_odd_N.query("train2003 != '*'")
even_Z_odd_N = even_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_odd_N = odd_Z_odd_N.assign(ids = pd.Series(np.array([2] * len(odd_Z_odd_N))))
odd_Z_odd_N = odd_Z_odd_N.query("train2003 != '*'")
odd_Z_odd_N = odd_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
even_Z_even_N = even_Z_even_N.assign(ids = pd.Series(np.array([3] * len(even_Z_even_N))))
even_Z_even_N = even_Z_even_N.query("train2003 != '*'")
even_Z_even_N = even_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N = odd_Z_even_N.assign(ids = pd.Series(np.array([4] * len(odd_Z_even_N))))
odd_Z_even_N = odd_Z_even_N.query("train2003 != '*'")
odd_Z_even_N = odd_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")


# In[11]:


model_list = ['SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24']
# Evidence integral calculation
for model in mode_list:
    evidence_integral(even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N ,model = model , domain = "ca", n_mc = 20000)


# In[35]:


# Dictionary with posterior samples of masses for each of the model with nuclei in AME2003
with open('posterior_masses_train.pkl', 'rb') as f:
     posteriors_dict = pickle.load(f)
# The BMA posterior sample for nuclei in AME2003
posterior_train(posteriors_dict, even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N , n_mc = 200000, domain = "ca")


# In[41]:


# The BMA posterior sample for nuclei in AME2016
# Preprocessing the data for a testing set evaluation
even_Z_odd_N = pd.read_csv("data_S1n_2018_even_Z_odd_N.csv")
even_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_odd_N = even_Z_odd_N.assign(tuples = list(zip(even_Z_odd_N["Z"], even_Z_odd_N["N"])))
odd_Z_odd_N = pd.read_csv("data_S1n_2018_odd_Z_odd_N.csv")
odd_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_odd_N = odd_Z_odd_N.assign(tuples = list(zip(odd_Z_odd_N["Z"], odd_Z_odd_N["N"])))
even_Z_even_N = pd.read_csv("data_S2n_2018_even_Z_even_N.csv")
even_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_even_N = even_Z_even_N.assign(tuples = list(zip(even_Z_even_N["Z"], even_Z_even_N["N"])))
odd_Z_even_N = pd.read_csv("data_S2n_2018_odd_Z_even_N.csv")
odd_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_even_N = odd_Z_even_N.assign(tuples = list(zip(odd_Z_even_N["Z"], odd_Z_even_N["N"])))

#Filtration
even_Z_odd_N = even_Z_odd_N.assign(ids = pd.Series(np.array([1] * len(even_Z_odd_N))))
even_Z_odd_N = even_Z_odd_N.query("test2016 != '*'")
even_Z_odd_N = even_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_odd_N = odd_Z_odd_N.assign(ids = pd.Series(np.array([2] * len(odd_Z_odd_N))))
odd_Z_odd_N = odd_Z_odd_N.query("test2016 != '*'")
odd_Z_odd_N = odd_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
even_Z_even_N = even_Z_even_N.assign(ids = pd.Series(np.array([3] * len(even_Z_even_N))))
even_Z_even_N = even_Z_even_N.query("test2016 != '*'")
even_Z_even_N = even_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N = odd_Z_even_N.assign(ids = pd.Series(np.array([4] * len(odd_Z_even_N))))
odd_Z_even_N = odd_Z_even_N.query("test2016 != '*'")
odd_Z_even_N = odd_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N = odd_Z_even_N.drop(147)

with open('posterior_masses_test.pkl', 'rb') as f:
     posteriors_dict = pickle.load(f)
posterior_test(posteriors_dict, even_Z_odd_N, odd_Z_odd_N, even_Z_even_N, odd_Z_even_N , n_mc = 200000, domain = "ca")


# In[5]:


# Results for AME2003
alpha = 1
model_list = ['SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24']
dataset_eval = "train2003"
with open('posterior_masses_train.pkl', 'rb') as f:
     posteriors_dict = pickle.load(f)
even_Z_odd_N = pd.read_csv("data_S1n_2018_even_Z_odd_N.csv")
even_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_odd_N = even_Z_odd_N.assign(tuples = list(zip(even_Z_odd_N["Z"], even_Z_odd_N["N"])))
odd_Z_odd_N = pd.read_csv("data_S1n_2018_odd_Z_odd_N.csv")
odd_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_odd_N = odd_Z_odd_N.assign(tuples = list(zip(odd_Z_odd_N["Z"], odd_Z_odd_N["N"])))
even_Z_even_N = pd.read_csv("data_S2n_2018_even_Z_even_N.csv")
even_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_even_N = even_Z_even_N.assign(tuples = list(zip(even_Z_even_N["Z"], even_Z_even_N["N"])))
odd_Z_even_N = pd.read_csv("data_S2n_2018_odd_Z_even_N.csv")
odd_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_even_N = odd_Z_even_N.assign(tuples = list(zip(odd_Z_even_N["Z"], odd_Z_even_N["N"])))

#Filtration
even_Z_odd_N = even_Z_odd_N.assign(ids = pd.Series(np.array([1] * len(even_Z_odd_N))))
even_Z_odd_N = even_Z_odd_N.query("train2003 != '*'")
even_Z_odd_N = even_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_odd_N = odd_Z_odd_N.assign(ids = pd.Series(np.array([2] * len(odd_Z_odd_N))))
odd_Z_odd_N = odd_Z_odd_N.query("train2003 != '*'")
odd_Z_odd_N = odd_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
even_Z_even_N = even_Z_even_N.assign(ids = pd.Series(np.array([3] * len(even_Z_even_N))))
even_Z_even_N = even_Z_even_N.query("train2003 != '*'")
even_Z_even_N = even_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N = odd_Z_even_N.assign(ids = pd.Series(np.array([4] * len(odd_Z_even_N))))
odd_Z_even_N = odd_Z_even_N.query("train2003 != '*'")
odd_Z_even_N = odd_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")

domain = "ca"
if domain == "ca":
    even_Z_odd_N = even_Z_odd_N[(even_Z_odd_N.Z <= 22) & (even_Z_odd_N.Z >= 14)]
    odd_Z_odd_N = odd_Z_odd_N[(odd_Z_odd_N.Z <= 22) & (odd_Z_odd_N.Z >= 14)]
    even_Z_even_N = even_Z_even_N[(even_Z_even_N.Z <= 22) & (even_Z_even_N.Z >= 14)]
    odd_Z_even_N = odd_Z_even_N[(odd_Z_even_N.Z <= 22) & (odd_Z_even_N.Z >= 14)]

dataset_eval = "train2003"
even_Z_odd_N_posterior = np.load('even_Z_odd_N_new__observable_eo_alpha_' + str(alpha) + "_train_20.npy")
odd_Z_odd_N_posterior = np.load('odd_Z_odd_N_new__observable_oo_alpha_' + str(alpha) + "_train_20.npy")
even_Z_even_N_posterior = np.load('even_Z_even_N_new__observable_ee_alpha_' + str(alpha) + "_train_20.npy")
odd_Z_even_N_posterior = np.load('odd_Z_even_N_new__observable_oe_alpha_' + str(alpha) + "_train_20.npy")
    
    
posterior_mse = {}
for model_name in model_list:
    ss_eo = np.sum((posteriors_dict['even_Z_odd_N_' + model_name].iloc[:,:-3].mean(axis=1).values - even_Z_odd_N[dataset_eval].astype(float).values) ** 2)
    ss_oo = np.sum((posteriors_dict['odd_Z_odd_N_' + model_name].iloc[:,:-3].mean(axis=1).values - odd_Z_odd_N[dataset_eval].astype(float).values) ** 2)
    ss_ee = np.sum((posteriors_dict['even_Z_even_N_' + model_name].iloc[:,:-3].mean(axis=1).values - even_Z_even_N[dataset_eval].astype(float).values) ** 2)
    ss_oe = np.sum((posteriors_dict['odd_Z_even_N_' + model_name].iloc[:,:-3].mean(axis=1).values - odd_Z_even_N[dataset_eval].astype(float).values) ** 2)
    posterior_mse[model_name] = (ss_eo + ss_oo + ss_ee + ss_oe) / (len(posteriors_dict['even_Z_odd_N_' + model_name]) +
                                                                  len(posteriors_dict['odd_Z_odd_N_' + model_name]) + 
                                                                  len(posteriors_dict['even_Z_even_N_' + model_name]) +
                                                                  len(posteriors_dict['odd_Z_even_N_' + model_name]))


BMA_ss_EO = np.sum((np.mean(even_Z_odd_N_posterior, axis = 1) - even_Z_odd_N[dataset_eval].astype(float).values) ** 2)
BMA_ss_OO = np.sum((np.mean(odd_Z_odd_N_posterior, axis = 1) - odd_Z_odd_N[dataset_eval].astype(float).values) ** 2)
BMA_ss_EE = np.sum((np.mean(even_Z_even_N_posterior, axis = 1) - even_Z_even_N[dataset_eval].astype(float).values) ** 2)
BMA_ss_OE = np.sum((np.mean(odd_Z_even_N_posterior, axis = 1) - odd_Z_even_N[dataset_eval].astype(float).values) ** 2)
BMA_mse = (BMA_ss_EO + BMA_ss_OO + BMA_ss_EE + BMA_ss_OE) / (len(even_Z_odd_N_posterior) + 
                                                                    len(odd_Z_odd_N_posterior) +
                                                                    len(even_Z_even_N_posterior) +
                                                                    len(odd_Z_even_N_posterior))

print('BMA root MSE : ' + str(np.sqrt(BMA_mse)))
print('Models root MSE: ')
print({k: np.round(np.sqrt(v), decimals= 3) for k,v in posterior_mse.items()})
print('r^2: ')
print({k: np.round(1 - BMA_mse / v, decimals=3) for k,v in posterior_mse.items()})


# In[6]:


# Results for AME2016 \ AME2003
model_list = ['SKM', 'SKP', 'SLY4', 'SVMIN', 'UNEDF0', 'UNEDF1', 'UNEDF2', 'FRDM2012', 'HFB24']
alpha = 1
dataset_eval = "train2003"
even_Z_odd_N = pd.read_csv("data_S1n_2018_even_Z_odd_N.csv")
even_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_odd_N = even_Z_odd_N.assign(tuples = list(zip(even_Z_odd_N["Z"], even_Z_odd_N["N"])))
odd_Z_odd_N = pd.read_csv("data_S1n_2018_odd_Z_odd_N.csv")
odd_Z_odd_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_odd_N = odd_Z_odd_N.assign(tuples = list(zip(odd_Z_odd_N["Z"], odd_Z_odd_N["N"])))
even_Z_even_N = pd.read_csv("data_S2n_2018_even_Z_even_N.csv")
even_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_even_N = even_Z_even_N.assign(tuples = list(zip(even_Z_even_N["Z"], even_Z_even_N["N"])))
odd_Z_even_N = pd.read_csv("data_S2n_2018_odd_Z_even_N.csv")
odd_Z_even_N.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_even_N = odd_Z_even_N.assign(tuples = list(zip(odd_Z_even_N["Z"], odd_Z_even_N["N"])))

#Filtration
even_Z_odd_N = even_Z_odd_N.assign(ids = pd.Series(np.array([1] * len(even_Z_odd_N))))
even_Z_odd_N = even_Z_odd_N.query("train2003 != '*'")
even_Z_odd_N = even_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_odd_N = odd_Z_odd_N.assign(ids = pd.Series(np.array([2] * len(odd_Z_odd_N))))
odd_Z_odd_N = odd_Z_odd_N.query("train2003 != '*'")
odd_Z_odd_N = odd_Z_odd_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
even_Z_even_N = even_Z_even_N.assign(ids = pd.Series(np.array([3] * len(even_Z_even_N))))
even_Z_even_N = even_Z_even_N.query("train2003 != '*'")
even_Z_even_N = even_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N = odd_Z_even_N.assign(ids = pd.Series(np.array([4] * len(odd_Z_even_N))))
odd_Z_even_N = odd_Z_even_N.query("train2003 != '*'")
odd_Z_even_N = odd_Z_even_N.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")

domain = "ca"
if domain == "ca":
    even_Z_odd_N = even_Z_odd_N[(even_Z_odd_N.Z <= 22) & (even_Z_odd_N.Z >= 14)]
    odd_Z_odd_N = odd_Z_odd_N[(odd_Z_odd_N.Z <= 22) & (odd_Z_odd_N.Z >= 14)]
    even_Z_even_N = even_Z_even_N[(even_Z_even_N.Z <= 22) & (even_Z_even_N.Z >= 14)]
    odd_Z_even_N = odd_Z_even_N[(odd_Z_even_N.Z <= 22) & (odd_Z_even_N.Z >= 14)]
    
# preprocessing the data for a testing set evaluation
dataset_eval = "test2016"
even_Z_odd_N_test = pd.read_csv("data_S1n_2018_even_Z_odd_N.csv")
even_Z_odd_N_test.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_odd_N_test = even_Z_odd_N_test.assign(tuples = list(zip(even_Z_odd_N_test["Z"], even_Z_odd_N_test["N"])))
odd_Z_odd_N_test = pd.read_csv("data_S1n_2018_odd_Z_odd_N.csv")
odd_Z_odd_N_test.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_odd_N_test = odd_Z_odd_N_test.assign(tuples = list(zip(odd_Z_odd_N_test["Z"], odd_Z_odd_N_test["N"])))
even_Z_even_N_test = pd.read_csv("data_S2n_2018_even_Z_even_N.csv")
even_Z_even_N_test.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
even_Z_even_N_test = even_Z_even_N_test.assign(tuples = list(zip(even_Z_even_N_test["Z"], even_Z_even_N_test["N"])))
odd_Z_even_N_test = pd.read_csv("data_S2n_2018_odd_Z_even_N.csv")
odd_Z_even_N_test.rename(columns={"2003":"train2003", "2016":"test2016", "SKM*": "SKM"}, inplace=True)
odd_Z_even_N_test = odd_Z_even_N_test.assign(tuples = list(zip(odd_Z_even_N_test["Z"], odd_Z_even_N_test["N"])))

#Filtration
even_Z_odd_N_test = even_Z_odd_N_test.assign(ids = pd.Series(np.array([1] * len(even_Z_odd_N_test))))
even_Z_odd_N_test = even_Z_odd_N_test.query("test2016 != '*'")
even_Z_odd_N_test = even_Z_odd_N_test.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_odd_N_test = odd_Z_odd_N_test.assign(ids = pd.Series(np.array([2] * len(odd_Z_odd_N_test))))
odd_Z_odd_N_test = odd_Z_odd_N_test.query("test2016 != '*'")
odd_Z_odd_N_test = odd_Z_odd_N_test.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
even_Z_even_N_test = even_Z_even_N_test.assign(ids = pd.Series(np.array([3] * len(even_Z_even_N_test))))
even_Z_even_N_test = even_Z_even_N_test.query("test2016 != '*'")
even_Z_even_N_test = even_Z_even_N_test.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N_test = odd_Z_even_N_test.assign(ids = pd.Series(np.array([4] * len(odd_Z_even_N_test))))
odd_Z_even_N_test = odd_Z_even_N_test.query("test2016 != '*'")
odd_Z_even_N_test = odd_Z_even_N_test.query("SKM != '*' and SKP != '*' and SLY4 != '*' and SVMIN != '*' and UNEDF0 != '*' and UNEDF1 != '*' and UNEDF2 != '*' and FRDM2012 != '*' and HFB24 != '*'")
odd_Z_even_N_test = odd_Z_even_N_test.drop(147)

if domain == "ca":
    even_Z_odd_N_test = even_Z_odd_N_test[(even_Z_odd_N_test.Z <= 22) & (even_Z_odd_N_test.Z >= 14)]
    odd_Z_odd_N_test = odd_Z_odd_N_test[(odd_Z_odd_N_test.Z <= 22) & (odd_Z_odd_N_test.Z >= 14)]
    even_Z_even_N_test = even_Z_even_N_test[(even_Z_even_N_test.Z <= 22) & (even_Z_even_N_test.Z >= 14)]
    odd_Z_even_N_test = odd_Z_even_N_test[(odd_Z_even_N_test.Z <= 22) & (odd_Z_even_N_test.Z >= 14)]

even_Z_odd_N_mask = np.logical_not(even_Z_odd_N_test["tuples"].isin(even_Z_odd_N["tuples"]))
odd_Z_odd_N_mask = np.logical_not(odd_Z_odd_N_test["tuples"].isin(odd_Z_odd_N["tuples"]))
even_Z_even_N_mask = np.logical_not(even_Z_even_N_test["tuples"].isin(even_Z_even_N["tuples"]))
odd_Z_even_N_mask = np.logical_not(odd_Z_even_N_test["tuples"].isin(odd_Z_even_N["tuples"]))

with open('posterior_masses_test.pkl', 'rb') as f:
     posteriors_dict = pickle.load(f)

even_Z_odd_N_posterior = np.load('even_Z_odd_N_new__observable_eo_alpha_' + str(alpha) + "_test_20.npy")
odd_Z_odd_N_posterior = np.load('odd_Z_odd_N_new__observable_oo_alpha_' + str(alpha) + "_test_20.npy")
even_Z_even_N_posterior = np.load('even_Z_even_N_new__observable_ee_alpha_' + str(alpha) + "_test_20.npy")
odd_Z_even_N_posterior = np.load('odd_Z_even_N_new__observable_oe_alpha_' + str(alpha) + "_test_20.npy")

even_Z_odd_N_posterior = even_Z_odd_N_posterior[even_Z_odd_N_mask.values,:]
odd_Z_odd_N_posterior = odd_Z_odd_N_posterior[odd_Z_odd_N_mask.values,:]
even_Z_even_N_posterior = even_Z_even_N_posterior[even_Z_even_N_mask.values,:]
odd_Z_even_N_posterior = odd_Z_even_N_posterior[odd_Z_even_N_mask.values,:]
even_Z_odd_N_test = even_Z_odd_N_test[even_Z_odd_N_mask.values]
odd_Z_odd_N_test = odd_Z_odd_N_test[odd_Z_odd_N_mask.values]
even_Z_even_N_test = even_Z_even_N_test[even_Z_even_N_mask.values]
odd_Z_even_N_test = odd_Z_even_N_test[odd_Z_even_N_mask.values]
posterior_mse = {}

for model_name in model_list:
    ss_eo = np.sum((posteriors_dict['even_Z_odd_N_' + model_name].iloc[:,:-3].mean(axis=1).values[even_Z_odd_N_mask] - even_Z_odd_N_test[dataset_eval].astype(float).values) ** 2)
    ss_oo = np.sum((posteriors_dict['odd_Z_odd_N_' + model_name].iloc[:,:-3].mean(axis=1).values[odd_Z_odd_N_mask] - odd_Z_odd_N_test[dataset_eval].astype(float).values) ** 2)
    ss_ee = np.sum((posteriors_dict['even_Z_even_N_' + model_name].iloc[:,:-3].mean(axis=1).values[even_Z_even_N_mask] - even_Z_even_N_test[dataset_eval].astype(float).values) ** 2)
    ss_oe = np.sum((posteriors_dict['odd_Z_even_N_' + model_name].iloc[:,:-3].mean(axis=1).values[odd_Z_even_N_mask] - odd_Z_even_N_test[dataset_eval].astype(float).values) ** 2)
    posterior_mse[model_name] = (ss_eo + ss_oo + ss_ee + ss_oe) / (len(posteriors_dict['even_Z_odd_N_' + model_name].values[even_Z_odd_N_mask]) +
                                                                  len(posteriors_dict['odd_Z_odd_N_' + model_name].values[odd_Z_odd_N_mask]) + 
                                                                  len(posteriors_dict['even_Z_even_N_' + model_name].values[even_Z_even_N_mask]) +
                                                                  len(posteriors_dict['odd_Z_even_N_' + model_name].values[odd_Z_even_N_mask]))
BMA_ss_EO = np.sum((np.mean(even_Z_odd_N_posterior, axis = 1) - even_Z_odd_N_test[dataset_eval].astype(float).values) ** 2)
BMA_ss_OO = np.sum((np.mean(odd_Z_odd_N_posterior, axis = 1) - odd_Z_odd_N_test[dataset_eval].astype(float).values) ** 2)
BMA_ss_EE = np.sum((np.mean(even_Z_even_N_posterior, axis = 1) - even_Z_even_N_test[dataset_eval].astype(float).values) ** 2)
BMA_ss_OE = np.sum((np.mean(odd_Z_even_N_posterior, axis = 1) - odd_Z_even_N_test[dataset_eval].astype(float).values) ** 2)
BMA_mse = (BMA_ss_EO + BMA_ss_OO + BMA_ss_EE + BMA_ss_OE) / (len(even_Z_odd_N_posterior) + 
                                                                len(odd_Z_odd_N_posterior) +
                                                                len(even_Z_even_N_posterior) +
                                                                len(odd_Z_even_N_posterior))

print('BMA root MSE : ' + str(np.sqrt(BMA_mse)))
print('Models root MSE: ')
print({k: np.round(np.sqrt(v), decimals= 3) for k,v in posterior_mse.items()})
print('r^2: ')
print({k: np.round(1 - BMA_mse / v, decimals=3) for k,v in posterior_mse.items()})

