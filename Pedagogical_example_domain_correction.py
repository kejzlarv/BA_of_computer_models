#!/usr/bin/env python
# coding: utf-8


# Loading the neccesary packages
import numpy as np
import pandas as pd
import pymc3 as pm 
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import re
import os
import sys
from tqdm import tqdm

import matplotlib as plt_base
plt_base.rcParams.update({'errorbar.capsize': 3}) # Seting caps on error bars



# Utilities
def sample_mixture(sample_M1, sample_M2, M1_to_M2_ratio):
    """Function sampel from the mixture under models M1 and M2 according to the provided ratio.
    
    Where:
    sample_M1 - sample from the distribution under M1
    sample_M2 - sample from the distribution under M2
    M1_to_M2_ratio - Ration of the samples for M1 in favor of M2
    """           

    i = 0
    j = 0
    mixture_sample = np.random.randn(sample_M1.shape[0])[:,None]
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

def correction_factor(x_diff, y_diff, x_intersection, y_intersection, n_mc = 40000):
    """Function calculates the correction factor for given model"""
    
    # The intersection model
    x_sq = x_intersection ** 2
    x_input = np.concatenate((x_intersection, x_sq), axis = 1)

    #MCMC model - correction factors and shif
    x_shared = theano.shared(x_input)
    gp_mean_coeff = np.array([0, epsilon, c])

    gamma_alpha = 1
    gamma_beta = 10

    inv_gamma_alpha = 1
    inv_gamma_beta = 10

    with pm.Model() as gp_posteriors_model:
        #Priors
        tau_sq = pm.InverseGamma("tau_sq", alpha = inv_gamma_alpha, beta = inv_gamma_beta)
        sigma_sq = pm.InverseGamma("sigma_sq", alpha = 10, beta= 1)
        lamb_sq = pm.Gamma("lamb_sq", alpha = gamma_alpha, beta = gamma_beta, shape = 2)
        theta = pm.Normal("theta", mu= 0, sd = 1)

        #Shared variables for the input
        x_input_theta = tt.concatenate([x_shared, tt.tile(theta, (len(x_input), 1))], axis = 1)

        #GP definition
        #Mean
        mean_gp = pm.gp.mean.Linear(coeffs = gp_mean_coeff, intercept = 0)
        #Covariance
        cov_gp = tau_sq * pm.gp.cov.ExpQuad(x_input.shape[1] + 1, ls = tt.sqrt(lamb_sq) / 4, active_dims = [0,2])
        #GP
        gp_model = pm.gp.Marginal(mean_func=mean_gp, cov_func= cov_gp)

        #Marginal likelihoods
        y_ = gp_model.marginal_likelihood("y_", X = x_input_theta, y = y_intersection, noise = tt.sqrt(sigma_sq))
        trace_priors = pm.sample(n_mc, tune = 10000, chains = 1)       
        
    # The complement model
    x_sq = x_diff ** 2
    x_input = np.concatenate((x_diff, x_sq), axis = 1)

    #MCMC model - correction factors and shif
    x_shared = theano.shared(x_input)
    gp_mean_coeff = np.array([0, epsilon, c])

    gamma_alpha = 1
    gamma_beta = 10

    inv_gamma_alpha = 1
    inv_gamma_beta = 10

    with pm.Model() as pymc3_model:
        #Priors
        tau_sq = pm.InverseGamma("tau_sq", alpha = inv_gamma_alpha, beta = inv_gamma_beta)
        sigma_sq = pm.InverseGamma("sigma_sq", alpha = 10, beta= 1)
        lamb_sq = pm.Gamma("lamb_sq", alpha = gamma_alpha, beta = gamma_beta, shape = 2)
        theta = pm.Normal("theta", mu= 0, sd = 1)

        #Shared variables for the input
        x_input_theta = tt.concatenate([x_shared, tt.tile(theta, (len(x_input), 1))], axis = 1)

        #GP definition
        #Mean
        mean_gp = pm.gp.mean.Linear(coeffs = gp_mean_coeff, intercept = 0)
        #Covariance
        cov_gp = tau_sq * pm.gp.cov.ExpQuad(x_input.shape[1] + 1, ls = tt.sqrt(lamb_sq) / 4, active_dims = [0,2])
        #GP
        gp_model = pm.gp.Marginal(mean_func=mean_gp, cov_func= cov_gp)

        #Marginal likelihoods
        y_gp = gp_model.marginal_likelihood("y_", X = x_input_theta, y = y_diff, noise = tt.sqrt(sigma_sq))    
    
    
    log_likelihood = np.empty(0)
    mc_integral = np.empty(n_mc)    
    logp = y_gp.logp

    for i in tqdm(range(n_mc), desc = "Log likelihood eval"):
        log_likelihood = np.append(log_likelihood, logp(trace_priors[i]))
    
    for i in tqdm(range(n_mc), desc = "Integral calc"):
        m = max(log_likelihood[:(i + 1)])
        mc_integral[i] = (np.exp(m) * np.sum(np.exp(log_likelihood[:(i + 1)] - m))) / (i + 1)   
    
    return log_likelihood, mc_integral

def posterior_predictions(pymc3_model, gp, posterior_trace, x_new, theta_dim, file, sn_start = 0, batch_n = 0,
                          batch_size = 500, n_core = 1, noise = True):
    """Samples from posterior predictive distribution.
    
    Generates samples from posterior predictive distribution for given input x_new and
    posterior samples of calibration parameters.
    
    Args:
      pymc3_model: A pymc3_model obejct defining the model for prediciton
      gp: A pymc3.gp.marginal object representing the GP model for observables
      posterior_samples: A pymc3 trace with posterir samples of calibration parameters
      x_new: Values of inputs for prediction
      theta_dim: A dimension of the calibration parameters
      n_core: Number of cores to be used for the computation
      noise: Should predictive noise be included?
      
    Return:
      A numpy array with samples from predictive distribution
      
    """
    
    # Batching
    sn_start = sn_start
    sn_end = np.minimum(sn_start + batch_size, len(posterior_trace))
    batch = batch_n
    # Exception catching
    
    while (sn_start < sn_end):
        try:
            #Unix shell comands for smooth parallel computing with theano
            #os.system('rm -rf ~/theano/')
            #os.system('rm -rf /tmp/kejzlarv/theano.NOBACKUP')
            #os.system('theano-cache clear')
            batch += 1
            print("Batch start index: " + str(sn_start))
            prediction_samples = np.empty((len(x_new), 0))
            posterior_samples = posterior_trace[sn_start:sn_end]

            # Batch sampling
            if n_core == 1:
                for sample_point in tqdm(posterior_samples, desc = "Sampling predictive distribution"):
                    y_new = sample_prediction(sample_point, x_new, theta_dim, noise)
                    if len(x_new) == 1: # Prediction for one vale handling
                        prediction_samples = np.append(prediction_samples, y_new)
                    else:
                        prediction_samples = np.concatenate((prediction_samples, y_new), axis = 1)
            elif n_core > 1:
                prediction_samples = Parallel(n_jobs = n_core)(delayed(sample_prediction)(sample_point, x_new, theta_dim, noise)
                                                              for sample_point in posterior_samples)
                # Needs to be reshaped
                prediction_samples = np.array(prediction_samples)
                prediction_samples = prediction_samples.reshape(prediction_samples.shape[0:2]).T

            sn_start = sn_end
            sn_end = np.minimum(sn_start + batch_size, len(posterior_trace))
            # Unix
            #np.save(file + "/batch_" + str(batch) + "s_end_" + str(sn_start), prediction_samples)
            # Windows
            np.save(file + "\\batch_" + str(batch) + "_s_end_" + str(sn_start), prediction_samples)
        except Exception:
            print("Exception raised and the batch was skipped")
            sn_start = sn_end
            sn_end = np.minimum(sn_start + batch_size, len(posterior_trace))
         
    return prediction_samples

def sample_prediction(sample_point, x_new, theta_dim, noise = True):
    """Generate single sample from posterior predictive distribution.
    """
    
    # Sample point for theta
    theta_sample_point = sample_point["theta"]
            
    # Exception handling for ValueError
    try:
        with gp_toy_model:
            input_new = np.concatenate([x_new, np.tile(theta_sample_point, (len(x_new), 1))], axis = 1)
            mu, cov = gp_model.predict(input_new, point = sample_point, pred_noise = noise)
            if input_new.shape[0] == 1: # Prediction for one value handling
                prediction_sample_point = pm.MvNormal.dist(mu = mu, cov = cov).random(1)
            else:
                prediction_sample_point = pm.MvNormal.dist(mu = mu, cov = cov).random(1)[:, None]
    except ValueError:
        print("Exception in inner loop")
        return
    return prediction_sample_point

def batch_to_ndarray(file):
    """Combine batches of samples from posterior predictive distribution into a single numpy.ndarray.
    
    Args:
      file: A string with the path to the folder with batches.
      
    Returns:
      A pandas DataFrame that is concatenation of all the batches in "file"
    """
    file_batch = os.listdir(file)
    file_batch.sort(key=lambda x: int(re.findall('\d+', x)[0]))
    posterior_sample = np.load(file + "\\" + file_batch[0])
    for batch in file_batch[1:]:
        #print(batch)
        new_batch = np.load(file + "\\" + batch)
        if new_batch.ndim > 1:
            posterior_sample = np.concatenate((posterior_sample, new_batch),
                                          axis = posterior_sample.ndim - 1)
    posterior_sample = pd.DataFrame(posterior_sample).dropna(axis=1)
    return posterior_sample



####### This code reproduces Figure 2 and Table 4 in "Bayesian averaging of computer models with domain
####### discrepancies: a nuclear physics perspective" from already precomputed results - asymetric scenario

np.random.seed(123)
# Data
y = np.repeat(0,18)
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise

# Predictors
x = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])[:,None]
    
#Names base
evidence_m1_name = "evidence_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
evidence_m2_name = "evidence_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"
corr_m1_name = "corr_factor_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
corr_m2_name = "corr_factor_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"
pred_m1_name = "pred_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
pred_m2_name = "pred_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"

folder_location = "pred_pedag_antisymmetric/"

#Dictionaries for posterior samples and numerical 
posterior_samples = {}
posterior_summary = {}
shift_max = 3
range_post = np.array(range(10)) + 4
#range_post = np.array(range(10))
#range_post = np.array(range(18))
for i in range(shift_max):
    # Loading neccessary files
    m1_e = np.load(folder_location + evidence_m1_name + str(i) + ".npy")
    m2_e = np.load(folder_location + evidence_m2_name + str(i) + ".npy")
    c1 = np.load(folder_location + corr_m1_name + str(i) + ".npy")
    c2 = np.load(folder_location + corr_m2_name + str(i) + ".npy")
    prediction_sample_1 = np.array(batch_to_ndarray(folder_location + pred_m1_name + str(i)))[:,5000:]
    prediction_sample_2 = np.array(batch_to_ndarray(folder_location + pred_m2_name + str(i)))[:,5000:]
    
    ratio = m1_e[-1] * c2[-1]/(m2_e[-1]*c1[-1])
    ratio_simple = m1_e[-1] * 1/(m2_e[-1]*1)
    posterior_BMA = sample_mixture(prediction_sample_1, prediction_sample_2, ratio)
    posterior_BMA_simple = sample_mixture(prediction_sample_1, prediction_sample_2, ratio_simple)
    
    prediction_sample_1 = prediction_sample_1[range_post,:]
    prediction_sample_2 = prediction_sample_2[range_post,:]
    posterior_BMA = posterior_BMA[range_post,:]
    posterior_BMA_simple = posterior_BMA_simple[range_post,:]
    mse_1 = mse(y[range_post], prediction_sample_1)
    mse_2 = mse(y[range_post], prediction_sample_2)
    mse_bma = mse(y[range_post], posterior_BMA)
    mse_bma_simple = mse(y[range_post], posterior_BMA_simple)
    
    # Printing results
    print("#### shift: " + str(i))
    print("Q: " + str(np.round(ratio, decimals = 2)))
    print("evidence M1 = " + str(m1_e[-1]))
    print("evidence M2 = " + str(m2_e[-1]))
    print("correction factor M1 = " + str(c2[-1]))
    print("correction factor M2 = " + str(c1[-1]))
    print("root PMSE(M1) = " + str(np.round(np.sqrt(mse_1), decimals=2)))
    print("root PMSE(M2) = " + str(np.round(np.sqrt(mse_2), decimals=2)))
    print("root PMSE(BMA)_Q0 = " + str(np.round(np.sqrt(mse_bma_simple), decimals=2)))
    print("root PMSE(BMA)_Q = " + str(np.round(np.sqrt(mse_bma), decimals=2)))
    print("r^2 M1: " + str(np.round(1 - mse_bma / mse_1, decimals= 3)))
    print("r^2 M2: " + str(np.round(1 - mse_bma / mse_2, decimals= 3)))
    
    
    posterior_samples["shift_" + str(i)] = {"m1":prediction_sample_1, "m2":prediction_sample_2, "bma":posterior_BMA,
                                           "bma_simple": posterior_BMA_simple}
    posterior_summary["shift_" + str(i)] = {"mse1":mse_1, "mse2":mse_2, "mseBMA":mse_bma, "ratio":ratio,
                                           "m1e":m1_e[-1], "m2e":m2_e[-1], "c1":c1[-1], "c2":c2[-1] }
    
d_shared = np.array([0.3,0.5,0.7])
plt.rcParams.update({'font.size': 14})
thickness_c = 0.8
thickness_l = 0.9
fig, ax = plt.subplots(shift_max, 1)
fig.tight_layout()
shift_max = 3
for i in range(shift_max):
    #row = (i // 2)
    #col = i % 2
    row = i
    col = 0
    #ax.append(fig.add_subplot(grid[row,col]))
    
    m1 = posterior_samples["shift_" + str(i)]["m1"].mean(axis= 1)
    m2 = posterior_samples["shift_" + str(i)]["m2"].mean(axis= 1)
    mbma = posterior_samples["shift_" + str(i)]["bma"].mean(axis= 1)
    mbma_simple = posterior_samples["shift_" + str(i)]["bma_simple"].mean(axis= 1)
    m1_sd = posterior_samples["shift_" + str(i)]["m1"].std(axis= 1)
    m2_sd = posterior_samples["shift_" + str(i)]["m2"].std(axis= 1)
    mbma_sd = posterior_samples["shift_" + str(i)]["bma"].std(axis= 1)
    mbma_sd_simple = posterior_samples["shift_" + str(i)]["bma_simple"].std(axis= 1)
    
    l0 = ax[i].errorbar(x[range_post] + 0.1, mbma_simple, yerr=mbma_sd_simple, fmt='*b', ecolor='b', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA(Q_0)}$')
    l1 = ax[i].errorbar(x[range_post] - 0.1, mbma, yerr=mbma_sd, fmt='*g', ecolor='g', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA(Q)}$')
    l2 = ax[i].errorbar(x[range_post] + 0.3, m1, yerr=m1_sd, fmt='^r', ecolor='r', capthick=thickness_c, elinewidth=thickness_l,label=r'$\mathcal{M}_{1}$')
    l3 = ax[i].errorbar(x[range_post] - 0.3, m2, yerr=m2_sd, fmt='.k', ecolor='black', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{2}$')
    ax[i].plot([(x[range_post] + 0.5).flatten(), (x[range_post] - 0.5).flatten()], [y[range_post],y[range_post]], 'k-', linestyle = "--", color = "black", linewidth = thickness_l)
    ax[i].axes.tick_params(labelsize = 12)
    ax[i].text(0.5, 0.9,r'$D_{shared} = $' + str(d_shared[i]), ha='center', transform=ax[i].transAxes)
    ax[i].set_xlim([-5.8, 5.8])

plt.legend((l0,l1,l2,l3),(r'$\mathcal{M}_{BMA(Q_0)}$', r'$\mathcal{M}_{BMA(Q)}$', r'$\mathcal{M}_{1}$', r'$\mathcal{M}_{2}$'),loc = 'lower center', bbox_to_anchor = (0,-0.01,1,1),
            bbox_transform = plt.gcf().transFigure, ncol=4)
fig.text(0.03, 0.5, r'$\hat{y}^*$', va='center', rotation='vertical')
fig.set_size_inches(7, 6)
plt.subplots_adjust(wspace=0.1, hspace=0) 
plt.savefig("Idealized_ratios_antisymmetric.pdf",dpi = 300,bbox_inches='tight')



####### This code reproduces Figure 4 and Table 7 in "Bayesian averaging of computer models with domain
####### discrepancies: a nuclear physics perspective" from already precomputed results - symmetric scenario

np.random.seed(123)
# Data
y = np.repeat(0,18)
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise

# Predictors
x = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])[:,None]
    
#Names base
evidence_m1_name = "evidence_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
evidence_m2_name = "evidence_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"
corr_m1_name = "corr_factor_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
corr_m2_name = "corr_factor_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"
pred_m1_name = "pred_pedag_domain_c_1_eps_0.5_dataset_L_shift_"
pred_m2_name = "pred_pedag_domain_c_1_eps_-0.5_dataset_R_shift_"

folder_location = "pred_pedag_symmetric/"

#Dictionaries for posterior samples and numerical 
posterior_samples = {}
posterior_summary = {}
shift_max = 4
range_post = np.array(range(10)) + 4
for i in range(shift_max):
    # Loading neccessary files
    m1_e = np.load(folder_location + evidence_m1_name + str(i) + ".npy")
    m2_e = np.load(folder_location + evidence_m2_name + str(i) + ".npy")
    c1 = np.load(folder_location + corr_m1_name + str(i) + ".npy")
    c2 = np.load(folder_location + corr_m2_name + str(i) + ".npy")
    prediction_sample_1 = np.array(batch_to_ndarray(folder_location + pred_m1_name + str(i)))[:,5000:]
    prediction_sample_2 = np.array(batch_to_ndarray(folder_location + pred_m2_name + str(i)))[:,5000:]
    
    ratio = m1_e[-1] * c2[-1]/(m2_e[-1]*c1[-1])
    ratio_simple = m1_e[-1] * 1/(m2_e[-1]*1)
    posterior_BMA = sample_mixture(prediction_sample_1, prediction_sample_2, ratio)
    posterior_BMA_simple = sample_mixture(prediction_sample_1, prediction_sample_2, ratio_simple)
    
    prediction_sample_1 = prediction_sample_1[range_post,:]
    prediction_sample_2 = prediction_sample_2[range_post,:]
    posterior_BMA = posterior_BMA[range_post,:]
    posterior_BMA_simple = posterior_BMA_simple[range_post,:]
    mse_1 = mse(y[range_post], prediction_sample_1)
    mse_2 = mse(y[range_post], prediction_sample_2)
    mse_bma = mse(y[range_post], posterior_BMA)
    mse_bma_simple = mse(y[range_post], posterior_BMA_simple)
    
    # Printing results
    print("#### shift: " + str(i))
    print("Q: " + str(np.round(ratio, decimals = 2)))
    print("evidence M1 = " + str(m1_e[-1]))
    print("evidence M2 = " + str(m2_e[-1]))
    print("correction factor M1 = " + str(c2[-1]))
    print("correction factor M2 = " + str(c1[-1]))
    print("root PMSE(M1) = " + str(np.round(np.sqrt(mse_1), decimals=2)))
    print("root PMSE(M2) = " + str(np.round(np.sqrt(mse_2), decimals=2)))
    print("root PMSE(BMA)_Q0 = " + str(np.round(np.sqrt(mse_bma_simple), decimals=2)))
    print("root PMSE(BMA)_Q = " + str(np.round(np.sqrt(mse_bma), decimals=2)))
    print("r^2 M1: " + str(np.round(1 - mse_bma / mse_1, decimals= 3)))
    print("r^2 M2: " + str(np.round(1 - mse_bma / mse_2, decimals= 3)))
    
    
    posterior_samples["shift_" + str(i)] = {"m1":prediction_sample_1, "m2":prediction_sample_2, "bma":posterior_BMA,
                                           "bma_simple": posterior_BMA_simple}
    posterior_summary["shift_" + str(i)] = {"mse1":mse_1, "mse2":mse_2, "mseBMA":mse_bma, "ratio":ratio,
                                           "m1e":m1_e[-1], "m2e":m2_e[-1], "c1":c1[-1], "c2":c2[-1] }
    
d_shared = np.array([0.2,0.4,0.6,0.8])
plt.rcParams.update({'font.size': 14})
thickness_c = 0.8
thickness_l = 0.9
fig, ax = plt.subplots(shift_max, 1)
fig.tight_layout()
shift_max = 4
for i in range(shift_max):
    #row = (i // 2)
    #col = i % 2
    row = i
    col = 0
    #ax.append(fig.add_subplot(grid[row,col]))
    
    m1 = posterior_samples["shift_" + str(i)]["m1"].mean(axis= 1)
    m2 = posterior_samples["shift_" + str(i)]["m2"].mean(axis= 1)
    mbma = posterior_samples["shift_" + str(i)]["bma"].mean(axis= 1)
    mbma_simple = posterior_samples["shift_" + str(i)]["bma_simple"].mean(axis= 1)
    m1_sd = posterior_samples["shift_" + str(i)]["m1"].std(axis= 1)
    m2_sd = posterior_samples["shift_" + str(i)]["m2"].std(axis= 1)
    mbma_sd = posterior_samples["shift_" + str(i)]["bma"].std(axis= 1)
    mbma_sd_simple = posterior_samples["shift_" + str(i)]["bma_simple"].std(axis= 1)
    
    l0 = ax[i].errorbar(x[range_post] + 0.1, mbma_simple, yerr=mbma_sd_simple, fmt='*b', ecolor='b', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA(Q_0)}$')
    l1 = ax[i].errorbar(x[range_post] - 0.1, mbma, yerr=mbma_sd, fmt='*g', ecolor='g', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA(Q)}$')
    l2 = ax[i].errorbar(x[range_post] + 0.3, m1, yerr=m1_sd, fmt='^r', ecolor='r', capthick=thickness_c, elinewidth=thickness_l,label=r'$\mathcal{M}_{1}$')
    l3 = ax[i].errorbar(x[range_post] - 0.3, m2, yerr=m2_sd, fmt='.k', ecolor='black', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{2}$')
    ax[i].plot([(x[range_post] + 0.5).flatten(), (x[range_post] - 0.5).flatten()], [y[range_post],y[range_post]], 'k-', linestyle = "--", color = "black", linewidth = thickness_l)
    ax[i].axes.tick_params(labelsize = 12)
    ax[i].text(0.5, 0.9,r'$D_{shared} = $' + str(d_shared[i]), ha='center', transform=ax[i].transAxes)
    ax[i].set_xlim([-5.8, 5.8])

plt.legend((l0,l1,l2,l3),(r'$\mathcal{M}_{BMA(Q_0)}$', r'$\mathcal{M}_{BMA(Q)}$', r'$\mathcal{M}_{1}$', r'$\mathcal{M}_{2}$'),loc = 'lower center', bbox_to_anchor = (0,0.02,1,1),
            bbox_transform = plt.gcf().transFigure, ncol=4)
fig.text(-0.015, 0.5, r'$\hat{y}^*$', va='center', rotation='vertical')
fig.set_size_inches(7, 9.5)
plt.subplots_adjust(wspace=0.1, hspace=0) 
plt.savefig("Idealized_ratios_symmetric.pdf",dpi = 300,bbox_inches='tight')



#####################################################################################
# A pedagogical example domain corrected version computation code - symmetric scenario preset
np.random.seed(123)
# Data
y = np.repeat(0,18)
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise

# Predictors
x = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])[:,None]

# Shift determines the Dshared factor
data_shift = 0

l_x_train = x[(0 + int(data_shift)):(10 + int(data_shift))]
r_x_train = x[(8 - int(data_shift)):(18 - int(data_shift))]
x_intersection = x[(8 - int(data_shift)):(10 +int(data_shift))]
    
l_y_train = y[(0 + int(data_shift)):(10 + int(data_shift))]
r_y_train = y[(8 - int(data_shift)):(18 - int(data_shift))]
y_intersection = y[(8 - int(data_shift)):(10 + int(data_shift))]

# L for dataset starting at x = -9 with 0 shif => Dshared = 0.2 (L is always related to M1 computations)
# R for dataset starting at x = -1 with 0 shift => Dshared = 0.2 (R is always related to M2 computations)
dataset_type = "L"
if dataset_type == "L":
    x_train_a = l_x_train
    x_train_b = r_x_train
    y_train_a = l_y_train
    y_train_b = r_y_train
    x_diff = np.setdiff1d(x_train_a,x_train_b)[:,None]
    y_diff = y[np.isin(x, x_diff).reshape(len(y))]
elif dataset_type == "R":
    x_train_a = r_x_train
    x_train_b = l_x_train
    y_train_a = r_y_train
    y_train_b = l_y_train
    x_diff = np.setdiff1d(x_train_a,x_train_b)[:,None]
    y_diff = y[np.isin(x, x_diff).reshape(len(y))]

#Model parameters
c = 1
# Epsilon = 0.5 for model 1
# Epsilon = - 0.5 for model 2
epsilon = 0.5
#prediction noise
noise = False
    
n_mc = 160000
n_sample = 40000
n_tune = 20000


#####################################################################################
# A pedagogical example domain corrected version computation code - asymmetric scenario preset
np.random.seed(123)
# Data
y = np.repeat(0,18)
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise

# Predictors
x = np.array([-9,-8,-7,-6,-5,-4,-3,-2,-1,1,2,3,4,5,6,7,8,9])[:,None]

# Shift determines the Dshared factor
data_shift = int(sys.argv[1])

l_x_train = x[(0 + int(data_shift)):(10 + int(data_shift))]
r_x_train = x[(7 - int(data_shift)):(17 - int(data_shift))]
x_intersection = x[(8 - int(data_shift)):(10 +int(data_shift))]
    
l_y_train = y[(0 + int(data_shift)):(10 + int(data_shift))]
r_y_train = y[(7 - int(data_shift)):(17 - int(data_shift))]
y_intersection = y[(7 - int(data_shift)):(9 + int(data_shift))]

# L for dataset starting at x = -9 with 0 shif => Dshared = 0.2 (L is always related to M1 computations)
# R for dataset starting at x = -1 with 0 shift => Dshared = 0.2 (R is always related to M2 computations)
dataset_type = sys.argv[2]
if dataset_type == "L":
    x_train_a = l_x_train
    x_train_b = r_x_train
    y_train_a = l_y_train
    y_train_b = r_y_train
    x_diff = np.setdiff1d(x_train_a,x_train_b)[:,None]
    y_diff = y[np.isin(x, x_diff).reshape(len(y))]
elif dataset_type == "R":
    x_train_a = r_x_train
    x_train_b = l_x_train
    y_train_a = r_y_train
    y_train_b = l_y_train
    x_diff = np.setdiff1d(x_train_a,x_train_b)[:,None]
    y_diff = y[np.isin(x, x_diff).reshape(len(y))]

#Model parameters
c = 1
# Epsilon = 0.5 for model 1
# Epsilon = - 0.5 for model 2
epsilon = float(sys.argv[3])
#prediction noise
noise = False
    
n_mc = 160000
n_sample = 40000
n_tune = 20000



#Input for the pymc3 model
x_sq = x_train_a ** 2
x_input = np.concatenate((x_train_a, x_sq), axis = 1)

x_shared = theano.shared(x_input)
gp_mean_coeff = np.array([0, epsilon, c])

# Model prior definitions
gamma_alpha = 1
gamma_beta = 10

inv_gamma_alpha = 1
inv_gamma_beta = 10

with pm.Model() as gp_toy_model:
    #Priors
    tau_sq = pm.InverseGamma("tau_sq", alpha = inv_gamma_alpha, beta = inv_gamma_beta)
    sigma_sq = pm.InverseGamma("sigma_sq", alpha = 10, beta= 1)
    lamb_sq = pm.Gamma("lamb_sq", alpha = gamma_alpha, beta = gamma_beta, shape = 2)
    theta = pm.Normal("theta", mu= 0, sd = 1)
    
    #Shared variables for the input
    x_input_theta = tt.concatenate([x_shared, tt.tile(theta, (len(x_input), 1))], axis = 1)
    
    #GP definition
    #Mean
    mean_gp = pm.gp.mean.Linear(coeffs = gp_mean_coeff, intercept = 0)
    #Covariance
    cov_gp = tau_sq * pm.gp.cov.ExpQuad(x_input.shape[1] + 1, ls = tt.sqrt(lamb_sq) / 4, active_dims = [0,2])
    #GP
    gp_model = pm.gp.Marginal(mean_func=mean_gp, cov_func= cov_gp)
    
    #Marginal likelihoods
    y_gp = gp_model.marginal_likelihood("y_", X = x_input_theta, y = y_train_a, noise = tt.sqrt(sigma_sq))




# MCMC sampling
with gp_toy_model:
    trace = pm.sample(n_sample, tune = n_tune, njobs=1)

name =  "pedag_domain_c_1_eps_" + str(epsilon) + "_dataset_" + dataset_type + "_shift_" + str(data_shift)
pm.backends.ndarray.save_trace(trace, directory = name)


# Evidence integral computaiton
np.random.seed(123)
with pm.Model() as priors_model:
    
    tau_sq = pm.InverseGamma("tau_sq", alpha = inv_gamma_alpha, beta = inv_gamma_beta)
    sigma_sq = pm.InverseGamma("sigma_sq", alpha = 10, beta= 1)
    lamb_sq = pm.Gamma("lamb_sq", alpha = gamma_alpha, beta = gamma_beta, shape = 2)
    theta = pm.Normal("theta", mu= 0, sd = 1)
    trace_priors = pm.sample(n_mc, tune = 10000, chains = 1)


log_likelihood = np.empty(0)
mc_integral = np.empty(n_mc)
logp = y_gp.logp

for i in tqdm(range(n_mc), desc = "Log likelihood eval"):
    log_likelihood = np.append(log_likelihood, logp(trace_priors[i], transform = None))

for i in tqdm(range(n_mc), desc = "Integral calc"):
    m = max(log_likelihood[:(i + 1)])
    mc_integral[i] = (np.exp(m) * np.sum(np.exp(log_likelihood[:(i + 1)] - m))) / (i + 1)
                
np.save("evidence_" + name, mc_integral)

# Sampling predictive distribution for given epsilon

x_sq = x ** 2
x_input = np.concatenate((x, x_sq), axis = 1)
os.mkdir(os.getcwd() + "/pred_" + name)
post = posterior_predictions(gp_toy_model, gp_model, trace, x_input, 1, os.getcwd() + "/pred_" + name, batch_size = 500, n_core = 1, noise = noise)



# Correction factor computation
np.random.seed(123)
likelihood_eval, evidence = correction_factor(x_diff,y_diff, x_intersection, y_intersection, n_mc=160000)
np.save("corr_factor_" + name, evidence)

