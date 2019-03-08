
# coding: utf-8

# In[63]:


# Loading the neccesary packages
import numpy as np
import pymc3 as pm 
import matplotlib.pyplot as plt
import theano
import theano.tensor as tt
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing as mp
import re
from tqdm import tqdm


# In[64]:


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


# In[99]:


####### This code reproduces Figure 1 and Table 1 in "Bayesian averaging of computer models with domain
####### discrepancies:  towards a nuclear physics perspective" from already precomputed results

np.random.seed(123)
#Data
y = np.array([0,0,0,0,0,0,0,0,0,0])
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise
#Predictors
x = np.array([-5,-4,-3,-2,-1,1,2,3,4,5])[:,None]

# Loading evidence integrals
m1_e = np.load("evidence_pedag_simple_epsilon_0.5.npy")
m2_e = np.load("evidence_pedag_simple_epsilon_-0.5.npy")
print("Posterior M1: " + str(np.round(m1_e[-1]/(m1_e[-1] + m2_e[-1]), decimals=2)))
print("Posterior M2: " + str(np.round(m2_e[-1]/(m1_e[-1] + m2_e[-1]), decimals=2)))

# Loading samples from posterior predictive distirbutions
m1_y = np.load("posterior_pedag_simple_epsilon_0.5.npy")
m2_y = np.load("posterior_pedag_simple_epsilon_-0.5.npy")
posterior_BMA = np.load("posterior_pedag_simple_bma.npy")
m1 = m1_y.mean(axis= 1)
m2 = m2_y.mean(axis= 1)
mbma = posterior_BMA.mean(axis= 1)

# Root MSE results
print("root PMSE(M1) = " + str(np.round(np.sqrt(mse(y, m1_y)), decimals=3)))
print("root PMSE(M2) = " + str(np.round(np.sqrt(mse(y, m2_y)), decimals=3)))
print("root PMSE(BMA) = " + str(np.round(np.sqrt(mse(y, posterior_BMA)), decimals=3)))

# r^2
print("r^2 M1: " + str(np.round(1 - mse(y, posterior_BMA) / mse(y, m1_y), decimals= 3)))
print("r^2 M2: " + str(np.round(1 - mse(y, posterior_BMA) / mse(y, m2_y), decimals= 3)))

# Figure 1
thickness_c = 0.8
thickness_l = 0.9
plt.figure()
#plt.subplot(211)
plt.errorbar(x, mbma, fmt='*g', ecolor='g', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA}$')
plt.errorbar(x + 0.3, m1, fmt='^r', ecolor='r', capthick=thickness_c, elinewidth=thickness_l,label=r'$\mathcal{M}_{1}$')
plt.errorbar(x - 0.3, m2, fmt='.k', ecolor='black', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{2}$')
#plt.axhline(linestyle = "--", color = "black", linewidth = thickness_l)
plt.xlabel("x")
plt.ylabel(r'$\hat{y}^*$')
#plt.ylim(-0.08,0.08)
plt.plot([np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) + 0.3, np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) - 0.3], [y,y], 'k-', linestyle = "--", color = "black", linewidth = thickness_l)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)


# In[47]:


# A pedagogical example simple version with no model discrepancies
np.random.seed(123)
#Data
y = np.array([0,0,0,0,0,0,0,0,0,0])
sigma_noise = 0.001
y = y + np.random.randn(len(y)) * sigma_noise
#Predictors
x = np.array([-5,-4,-3,-2,-1,1,2,3,4,5])[:,None]

#Model parameters
c = 1
# Epsilon = 0.5 for model 1
# Epsilon = - 0.5 for model 2
epsilon = 0.5
noise = False   
    
n_mc = 100000
n_sample = 80000

#Input for the pymc3 model
x_sq = x ** 2
x_input = np.concatenate((x, x_sq), axis = 1)

x_shared = theano.shared(x_input)
gp_mean_coeff = np.array([0, epsilon, c])

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
    y_gp = gp_model.marginal_likelihood("y_", X = x_input_theta, y = y, noise = tt.sqrt(sigma_sq))



# MCM sampling
with gp_toy_model:
    #trace = pm.backends.ndarray.load_trace("pedag_simple_epsilon_" + str(epsilon))
    trace = pm.sample(n_sample, tune = 20000, njobs=1)

name =  "pedag_simple_epsilon_" + str(epsilon)
pm.backends.ndarray.save_trace(trace, directory = name)

# Evidence integral computaiton
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
    log_likelihood = np.append(log_likelihood, logp(trace_priors[i]))

for i in tqdm(range(n_mc), desc = "Integral calc"):
    m = max(log_likelihood[:(i + 1)])
    mc_integral[i] = (np.exp(m) * np.sum(np.exp(log_likelihood[:(i + 1)] - m))) / (i + 1)
                
np.save("evidence_pedag_simple_epsilon_" + str(epsilon), mc_integral)

# Sampling predictive distribution for given epsilon
os.mkdir(os.getcwd() + "/pred_pedag_simple_epsilon_" + str(epsilon))
post = posterior_predictions(gp_toy_model, gp_model, trace, x_input , 1, os.getcwd() + "/pred_pedag_simple_epsilon_" + str(epsilon), batch_size = 500, n_core = 1, noise = noise)


# In[82]:


# Posterior mean predictor results
np.random.seed(123)
m1_y = np.array(batch_to_ndarray("pred_pedag_simple_epsilon_0.5"))[:,20000:] #Burning samples first 20K samples
m2_y = np.array(batch_to_ndarray("pred_pedag_simple_epsilon_-0.5"))[:,20000:] #Burning samples first 20K samples
m1_e = np.load("evidence_pedag_simple_epsilon_0.5.npy")
m2_e = np.load("evidence_pedag_simple_epsilon_-0.5.npy")
ratio = m1_e[-1]/m2_e[-1]
posterior_BMA = sample_mixture(m1_y, m2_y, ratio)


# In[83]:


m1 = m1_y.mean(axis= 1)
m2 = m2_y.mean(axis= 1)
mbma = posterior_BMA.mean(axis= 1)


# In[54]:


m1_e = np.load("evidence_pedag_simple_epsilon_0.5.npy")
m2_e = np.load("evidence_pedag_simple_epsilon_-0.5.npy")
print("Posterior M1: " + str(np.round(m1_e[-1]/(m1_e[-1] + m2_e[-1]), decimals=2)))
print("Posterior M2: " + str(np.round(m2_e[-1]/(m1_e[-1] + m2_e[-1]), decimals=2)))


# In[89]:


# Root MSE results
print("root PMSE(M1) = " + str(np.round(np.sqrt(mse(y, m1_y)), decimals=3)))
print("root PMSE(M2) = " + str(np.round(np.sqrt(mse(y, m2_y)), decimals=3)))
print("root PMSE(BMA) = " + str(np.round(np.sqrt(mse(y, posterior_BMA)), decimals=3)))

# r^2
print("r^2 M1: " + str(np.round(1 - mse(y, posterior_BMA) / mse(y, m1_y), decimals= 3)))
print("r^2 M2: " + str(np.round(1 - mse(y, posterior_BMA) / mse(y, m2_y), decimals= 3)))


# In[90]:


thickness_c = 0.8
thickness_l = 0.9
plt.figure()
#plt.subplot(211)
plt.errorbar(x, mbma, fmt='*g', ecolor='g', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{BMA}$')
plt.errorbar(x + 0.3, m1, fmt='^r', ecolor='r', capthick=thickness_c, elinewidth=thickness_l,label=r'$\mathcal{M}_{1}$')
plt.errorbar(x - 0.3, m2, fmt='.k', ecolor='black', capthick=thickness_c, elinewidth=thickness_l, label=r'$\mathcal{M}_{2}$')
#plt.axhline(linestyle = "--", color = "black", linewidth = thickness_l)
plt.xlabel("x")
plt.ylabel(r'$\hat{y}^*$')
#plt.ylim(-0.08,0.08)
plt.plot([np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) + 0.3, np.array([-5,-4,-3,-2,-1,1,2,3,4,5]) - 0.3], [y,y], 'k-', linestyle = "--", color = "black", linewidth = thickness_l)
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=3, mode="expand", borderaxespad=0.)
plt.savefig("Idealized_no.png",dpi = 200)

