#!/usr/bin/env python
# coding: utf-8

# ### Packages and Function Definitoins

# In[1]:


# Loading neccesary packages
import numpy as np
import pymc3 as pm
import pandas as pd
import matplotlib.pyplot as plt
import pickle as pkl
import theano
import sys
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import scipy as sp

import seaborn as sns


# In[2]:


# Functions and utilities
def WS_potential(r, V0 = 50, a = 0.5, r0 = 1.25, A = 64):
    """Function returns value of WS potential for given r"""
    R = r0 * A ** (1 / 3)
    V_r = - V0 / (1 + np.exp((r - R) / a))
    return V_r

def C_potential(r, V0 = 50):
    """Function return value of C potential for given r"""
    return - V0 / r

def exp_data(r, sd, V0 = 50, a = 0.5, r0 = 1.25, A = 64):
    """Function generates observations as WS_potential / 2 + C_potential / 2 + sd * epsilon"""
    exp = WS_potential(r, V0, a, r0, A) * 0.5 + C_potential(r, V0) * 0.5 + sd * np.random.randn(len(r))
    return exp

def r_exp(n_exp = 50, r_range = [0.1, 10], domain_type = 1):
    """Generator of inputs"""
    if domain_type == 1:
        r = np.random.uniform(low = r_range[0], high = r_range[1], size =  n_exp)
    if domain_type == 2:
        r_1 = np.random.uniform(low = r_range[0] + 3, high = r_range[1], size =  int(n_exp / 2))
        r_2 = np.random.uniform(low = r_range[0], high = r_range[0] + 3, size =  int(n_exp / 2))
        r = np.concatenate((r_2,r_1), axis = 0)
        
    return r

def exp_model(r, y_exp, priors, model_type, V0 = 50, a = 0.5, r0 = 1.25, A = 64):
    """PYMC3 model definition for eithr WS or C model"""
    
    with pm.Model() as pm_model:

        # priors
        sigma = pm.InverseGamma("sigma", alpha = priors["sigma"][0], beta = priors["sigma"][1], transform=None)

        # covariance
        cov = pm.gp.cov.WhiteNoise(sigma)
        K = cov(r[:, None])

        # mean
        if model_type == "WS":
            m = WS_potential(r, V0, a, r0, A)
        if model_type == "C":
            m = C_potential(r, V0)

        # observations
        obs = pm.MvNormal('obs', mu = m, cov = K, observed = y_exp, transform=None)

    logp = obs.logp
    return pm_model, obs, logp

def evidence_int(priors, pm_model, obs, logp, n_mc):
    """MCMC approximation of evidence integral"""
    
    # evidence integral
    with pm.Model() as priors_model:
        # priors
        sigma = pm.InverseGamma("sigma", alpha = priors["sigma"][0], beta = priors["sigma"][1], transform=None)
        trace_priors = pm.sample(n_mc, tune = 5000, chains = 1)
    plt.hist(trace_priors["sigma"][trace_priors["sigma"] < 100])
    plt.show()
    
    log_likelihood = np.empty(0)
    mc_integral = np.empty(n_mc)
    for i in tqdm(range(n_mc), desc = "Log likelihood eval"):
        log_likelihood = np.append(log_likelihood, logp(trace_priors[i], transfrom = None))

    for i in tqdm(range(n_mc), desc = "Integral calc"):
        m = max(log_likelihood[:(i + 1)])
        mc_integral[i] = (np.exp(m) * np.sum(np.exp(log_likelihood[:(i + 1)] - m))) / (i + 1)
    plt.plot(mc_integral)
    plt.show()
    return mc_integral

def sample_predictions(r_new,  pm_model, obs, model_type, n_pred, V0 = 50, a = 0.5, r0 = 1.25, A = 64):
    """Sampler from posterior predictive distribution"""
    with pm_model:
        trace = pm.sample(n_pred, tune = 5000, chains = 1) 
        
    posterior_predictive = np.empty((len(r_new), 1))
    for sigma in tqdm(trace["sigma"], desc = "Posterior predictive distribution"):
        # Covariance
        cov = np.identity(len(r_new)) * sigma
        # Mean
        if model_type == "WS":
            mean = WS_potential(r_new, V0, a, r0, A)
        if model_type == "C":
            mean = C_potential(r_new, V0)
        # Sample from predictive distribution
        sample = np.random.multivariate_normal(mean, cov)[:,None]
        posterior_predictive = np.concatenate((posterior_predictive, sample), axis = 1)
            
    return posterior_predictive[:,1:], trace

def sample_mixture(sample_M1, sample_M2, M1_to_M2_ratio):
    """Function sampel from the mixture under models M1 and M2 according to the provided ratio.
    
    Where:
    sample_M1 - sample from the distribution under M1
    sample_M2 - sample from the distribution under M2
    M1_to_M2_ratio - Ration of the samples for M1 in favor of M2
    """           

    i = 0
    j = 0
    mixture_sample = np.empty((sample_M1.shape[0], 1))
    while (i < len(sample_M1.T)) and (j < len(sample_M2.T)):
        u = np.random.uniform()
        if u < (M1_to_M2_ratio / (M1_to_M2_ratio + 1)):
            sample_new = sample_M1[:,i]  
            mixture_sample = np.concatenate((mixture_sample, sample_new[:,None]), axis = 1)
            i += 1
        else:
            sample_new = sample_M2[:,j] 
            mixture_sample = np.concatenate((mixture_sample, sample_new[:,None]), axis = 1)
            j+=1
        
    return mixture_sample[:,1:]

def mse(Y_truth, Y_new):
    """Function calculates the mean squared error for the posterior samples Y_new"""
    Y_new_mean = np.mean(Y_new, axis = 1)[:,None]
    mse_calc = np.mean((Y_new_mean - Y_truth[:,None]) ** 2)
    return mse_calc

def credible_interval(data, alpha, ci_type = 'hpd'):
    """Function calculates bayesian credible intervals of various kinds
    
    Args:
        data: Numpy array of posterior samples (each row corrsponds to one obs)
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

def ECP(data, truth, alpha, ci_type):
    """Function calculates ECP for given significance level level"""
    
    counter = 0
    interval = credible_interval(data, alpha, ci_type = ci_type)
    for i in range(len(truth)):
        counter = counter + int((truth[i] >= interval[i,0]) and (truth[i] <= interval[i,1]))
        
    return counter / len(truth)


# ### Data generation and sampling
# #### Experimental data


np.random.seed(0)
########### This sets up the simulation
# n_exp is the size of experimental data
# r_range is the range of r values we want to consider
# eps is the sd of experimental error
# A is the atomic mass number of the nuclei under consideration

n_exp = 84
r_range = [0.1, 10]
eps = 1 # Experimental eror
A = 96
############

r = r_exp(n_exp = n_exp, r_range = r_range, domain_type = 1)
y_exp = exp_data(r, eps, A = A)

# Train/test split
train_size = int(len(y_exp) * 2 / 3)
train_sample = np.random.choice(len(y_exp), train_size, replace=False)
test_sample = list(set(np.arange(len(y_exp))) - set(train_sample))


r_train = r[train_sample]
y_train = y_exp[train_sample]

r_test = r[test_sample]
y_test = y_exp[test_sample]

# ### WS Model - Evidence 


np.random.seed(0)
priors = {"sigma": [1, 30]}
n_mc = 20000
pm_model_WS, obs_WS, logp_WS = exp_model(r_train, y_train, priors, model_type = "WS", A = A)
mc_integral_WS = evidence_int(priors, pm_model_WS, obs_WS, logp_WS, n_mc)


# ### C Model - Evidence

np.random.seed(0)
priors = {"sigma": [1, 30]}
n_mc = 20000
pm_model_C, obs_C, logp_C = exp_model(r_train, y_train, priors, model_type = "C", A = A)
mc_integral_C = evidence_int(priors, pm_model_C, obs_C, logp_C, n_mc)


print("P(M2|D) = " + str(mc_integral_C[-1] / (mc_integral_WS[-1] + mc_integral_C[-1])))
print("P(M1|D) = " + str(mc_integral_WS[-1] / (mc_integral_WS[-1] + mc_integral_C[-1])))


# ### Predictions
# setup
n_pred = 30000
r_new = r
M1_M2_ratio = mc_integral_WS[-1] / mc_integral_C[-1]


# #### WS Model - Prediction

np.random.seed(0)
pm_model_WS, obs_WS, logp_WS = exp_model(r, y_exp, priors, model_type = "WS", A = A)
predictions_WS, trace_WS = sample_predictions(r_new = r_new, pm_model = pm_model_WS, obs = obs_WS, model_type ="WS",
                                              n_pred = n_pred, A = A)


# #### C Model - Prediction

np.random.seed(0)
pm_model_C, obs_C, logp_C = exp_model(r, y_exp, priors, model_type = "C", A = A)
predictions_C, trace_C = sample_predictions(r_new = r_new, pm_model = pm_model_C, obs = obs_C, model_type ="C",
                                              n_pred = n_pred, A = A)


# #### BMA - Predictions
np.random.seed(0)
predictions_BMA = sample_mixture(predictions_C, predictions_WS, M1_M2_ratio)


print("RMSE BMA:" + str(np.sqrt(mse(y_exp[test_sample], predictions_BMA[test_sample,:]))))
print("RMSE W-S:" + str(np.sqrt(mse(y_exp[test_sample], predictions_WS[test_sample,:]))))
print("RMSE C:" + str(np.sqrt(mse(y_exp[test_sample], predictions_C[test_sample,:]))))

# r^2 for C potential
print("r^2: " + str(1 - mse(y_exp[test_sample], predictions_BMA[test_sample,:]) / np.min([mse(y_exp[test_sample], predictions_C[test_sample,:]), mse(y_exp[test_sample], predictions_C[test_sample,:])])))


# r^2 for WS potential


print("r^2: " + str(1 - mse(y_exp[test_sample], predictions_BMA[test_sample,:]) / np.min([mse(y_exp[test_sample], predictions_WS[test_sample,:]), mse(y_exp[test_sample], predictions_WS[test_sample,:])])))

# ECP

alpha_array = np.linspace(0.05, 1, 20) # Array of alphas
BMA_ECP = {}
C_ECP = {}
WS_ECP = {}
truth = y_exp[test_sample]
ci_type = 'et'
for alpha in tqdm(alpha_array , desc = "ECP for alpha"):
    BMA_ECP[str(alpha)] = ECP(predictions_BMA[test_sample,:], truth, alpha, ci_type)
    WS_ECP[str(alpha)] = ECP(predictions_WS[test_sample,:], truth, alpha, ci_type)
    C_ECP[str(alpha)] = ECP(predictions_C[test_sample,:], truth, alpha, ci_type)



plt.figure(figsize=(10,5))
x = 1 - alpha_array
plt.plot(x,x, 'r--' , linewidth = 1, label = 'Reference')
plt.plot(x,C_ECP.values(), marker = 'o', label = r'$\mathcal{M}_2$', linewidth = 1, markersize=5)
plt.plot(x,WS_ECP.values(), marker = '^', label = r'$\mathcal{M}_1$', linewidth = 1, markersize=5)
plt.plot(x,BMA_ECP.values(), c = 'k', marker = '*', linewidth = 1, markersize=5, label = r'$\mathcal{M}_{BMA}$')
plt.ylabel('Empirical coverage')
plt.xlabel('Credibility level')
plt.legend()
plt.savefig('ECP_WS_et_' + str(A) + "_nexp_" + str(n_exp) +".pdf", bbox_inches='tight',dpi = 300)
plt.show()

        

