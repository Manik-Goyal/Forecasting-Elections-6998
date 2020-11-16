import numpy as np
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

from pyro.infer import MCMC, NUTS, Predictive
from scipy.special import expit

def model(data, polls):
    #data from data dictionary

    #X
    N_national_polls = data["N_national_polls"] #Number of National Polls
    N_state_polls = data["N_state_polls"] #Number of State Polls
    T = data["T"] #Number of days
    S = data["S"] #Number of states for which at-least 1 poll is available
    P = data["P"] #Number of pollsters
    M = data["M"] #Number of poll modes
    Pop = data["Pop"] #Number of poll populations
    state = data["state"] #state index
    day_state = data["day_state"] #State Day index
    day_national = data["day_national"] #National Day index
    poll_state = data["poll_state"] #State Pollster Index
    poll_national = data["poll_national"] #National Pollster Index
    poll_mode_state = data["poll_mode_state"] #State Poll Mode Index
    poll_mode_national = data["poll_mode_national"] #National Poll Model Index
    poll_pop_state = data["poll_pop_state"] #State poll population
    poll_pop_national = data["poll_pop_national"] #National Poll Populaiton
    unadjusted_national = data["unadjusted_national"] 
    unadjusted_state = data["unadjusted_state"] 
    n_two_share_national = data["n_two_share_national"] #Total Number of Dem+Reb supporters for a particular poll
    n_two_share_state = data["n_two_share_state"] #Total Number of Dem+Reb supporters for a particular poll
    
    #y
    if polls is not None:
        n_democrat_national = polls["n_democrat_national"] #Number of Dem supporters in national poll for a particular poll 
        n_democrat_state = polls["n_democrat_state"] #Number of Dem supporters in state poll for a particular poll
    else:
        n_democrat_national = None
        n_democrat_state = None
    
    #Prior Input values
    mu_b_prior = data["mu_b_prior"]
    state_weights = data["state_weights"]
    sigma_c = data["sigma_c"]
    sigma_m = data["sigma_m"]
    sigma_pop = data["sigma_pop"]
    sigma_measure_noise_national = data["sigma_measure_noise_national"]
    sigma_measure_noise_state = data["sigma_measure_noise_state"]
    sigma_e_bias = data["sigma_e_bias"]

    #Covariance Matrix and Scale Input
    state_covariance_0 = data["state_covariance_0"]
    random_walk_scale = data["random_walk_scale"]
    mu_b_T_scale = data["mu_b_T_scale"]
    polling_bias_scale = data["polling_bias_scale"]

    #Data Transformation
    national_cov_matrix_error_sd = torch.sqrt(state_weights.T @ state_covariance_0 @ state_weights)

    #Scale Covariance
    ss_cov_poll_bias = state_covariance_0 * (polling_bias_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_T = state_covariance_0 * (mu_b_T_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_walk = state_covariance_0 * (random_walk_scale/national_cov_matrix_error_sd)**2

    #Cholesky Transformation
    cholesky_ss_cov_poll_bias = torch.cholesky(ss_cov_poll_bias)
    cholesky_ss_cov_mu_b_T = torch.cholesky(ss_cov_mu_b_T)
    cholesky_ss_cov_mu_b_walk = torch.cholesky(ss_cov_mu_b_walk)

    #Priors
    #Parameters
    with pyro.plate("raw_mu_b_T-plate", size = S):
        raw_mu_b_T = pyro.sample("mu_b_T", dist.Normal(0., 1.))
        assert raw_mu_b_T.shape == (S,)

    with pyro.plate("raw_mu_b_x-asis", size = S):
        with pyro.plate("raw_mu_b_y-plate", size = T):
           raw_mu_b = pyro.sample("mu_b_T", dist.Normal(0., 1.)) 
           raw_mu_b.t().flatten() #Matrix to Column Order Vector
    
    with pyro.plate("raw_mu_c-plate", size = P):
        raw_mu_c = pyro.sample("raw_mu_c", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_m-plate", size = M):
        raw_mu_m = pyro.sample("mu_m", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_pop-plate", size = Pop):
        raw_mu_pop = pyro.sample("raw_mu_pop", dist.Normal(0., 1.))

    #!Not sure if this satisfies Offset=0 and multiplier=0.02
    mu_e_bias = pyro.sample("mu_e_bias", dist.Normal(0., 0.02))*0.02 

    #!Need to find way to add constraint lower = 0, upper = 1
    rho_e_bias = pyro.sample("rho_e_bias", dist.Normal(0.7, 0.1)) 

    with pyro.plate("raw_e_bias-plate", size = T):
        raw_e_bias = pyro.sample("raw_e_bias", dist.Normal(0., 1.))

    with pyro.plate("raw_measure_noise_national-plate", size = N_national_polls):
        raw_measure_noise_national = pyro.sample("measure_noise_national", dist.Normal(0., 1.))
    
    with pyro.plate("raw_measure_noise_state-plate", size = N_state_polls):
        raw_measure_noise_state = pyro.sample("measure_noise_state", dist.Normal(0., 1.))

    with pyro.plate("raw_polling_bias-plate", size = S):
        raw_polling_bias = pyro.sample("polling_bias", dist.Normal(0., 1.))

    #Transformed Parameters
    mu_b = pyro.param('mu_b', torch.empty(S, T)) #initalize mu_b
    mu_b[:,T] = cholesky_ss_cov_mu_b_T @ raw_mu_b_T + mu_b_prior
    for i in range(1,T):
        mu_b[:, T - i] = cholesky_ss_cov_mu_b_walk @ raw_mu_b[:, T - i] + mu_b[:, T + 1 - i]
    
    mu_c = pyro.param('mu_c', raw_mu_c * sigma_c)
    mu_m = pyro.param('mu_m', raw_mu_m * sigma_m)
    mu_pop = pyro.param('mu_pop', raw_mu_pop * sigma_pop)
    sigma_rho = pyro.param('sigma_rho', torch.sqrt(1-(rho_e_bias)**2) * sigma_e_bias)

    e_bias = pyro.param('e_bias', torch.empty(T)) #initalize e_bias
    e_bias[1] = raw_e_bias[1] * sigma_e_bias
    for t in range(2,T+1):
        e_bias[t] = mu_e_bias + rho_e_bias * (e_bias[t - 1] - mu_e_bias) + raw_e_bias[t] * sigma_rho

    polling_bias = pyro.param('polling_bias', cholesky_ss_cov_poll_bias @ raw_polling_bias)
    national_mu_b_average = pyro.param('national_mu_b_average', mu_b.t() @ state_weights)
    national_polling_bias_average = pyro.param('national_polling_bias_average', polling_bias.T @ state_weights)

    logit_pi_democrat_state = pyro.param('logit_pi_democrat_state', torch.empty(N_state_polls))
    logit_pi_democrat_national = pyro.param('logit_pi_democrat_national', torch.empty(N_national_polls))

    for i in range(1,N_state_polls+1):
        logit_pi_democrat_state[i] = mu_b[state[i], day_state[i]] + mu_c[poll_state[i]] + mu_m[poll_mode_state[i]] + \
            mu_pop[poll_pop_state[i]] + unadjusted_state[i] * e_bias[day_state[i]] + raw_measure_noise_state[i] * sigma_measure_noise_state + \
            polling_bias[state[i]]
  
    logit_pi_democrat_national = national_mu_b_average[day_national] +  mu_c[poll_national] + mu_m[poll_mode_national] + \
        mu_pop[poll_pop_national] + unadjusted_national * e_bias[day_national] + raw_measure_noise_national * sigma_measure_noise_national +\
        national_polling_bias_average
    
    #Likelihood Of the Model
    #!need to verify if this is the correct implementation for binomial_logit of stan
    with pyro.plate("state-data-plate", size = N_state_polls):
        pyro.sample("n_democrat_state", dist.Binomial(n_two_share_state, logits = logit_pi_democrat_state), obs = n_democrat_state)
    
    with pyro.plate("national-data-plate", size = N_national_polls):
        pyro.sample("n_democrat_national", dist.Binomial(n_two_share_national, logits = logit_pi_democrat_national), obs = n_democrat_national)


#!Need to modify this based on hyper-parameters used by the Original Model
def Inference_MCMC(model, data, polls, n_samples = 500, n_warmup = 500, n_chains = 6):
    nuts_kernel = NUTS(model)
    mcmc = MCMC(nuts_kernel, num_samples = n_samples, warmup_steps = n_warmup, num_chains = n_chains)
    mcmc.run(data, polls)
    posterior_samples = mcmc.get_samples()

    hmc_samples = {k: v.detach().cpu().numpy() for k, v in mcmc.get_samples().items()}

    return posterior_samples, hmc_samples

#Generate samples from posterior predictive distribution
def sample_posterior_predictive(model, posterior_samples, n_samples, data):
    posterior_predictive = Predictive(model, posterior_samples, num_samples = n_samples)
    posterior_predictive_samples = posterior_predictive.get_samples(data, None)

    return posterior_predictive_samples

#Generating Quantity from posterior sample
def predicted_score(model, posterior_samples, data):
    T = data['T']
    S = data['S']
    predicted = torch.empty(T, S)
    posterior_predictive = Predictive(model, posterior_samples)
    trace  = posterior_predictive.get_vectorized_trace(data)

    mu_b = trace.nodes['mu_b']
    for s in range(1,S+1):
        predicted[:,s] = expit(mu_b[s,:].t().flatten())

    return predicted


