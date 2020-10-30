import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.distributions import constraints

import pyro
import pyro.distributions as dist
import pyro.optim as optim

from pyro.infer import EmpiricalMarginal, SVI, Trace_ELBO, Predictive
from pyro.contrib.autoguide import AutoMultivariateNormal
from pyro.infer.mcmc.api import MCMC
from pyro.infer.mcmc import NUTS


def model():
    raw_mu_b_T = pyro.sample("mu_b_T", dist.Normal(0., 1.))
    raw_mu_c = pyro.sample("mu_c", dist.Normal(0., 1.))
    raw_mu_m = pyro.sample("mu_m", dist.Normal(0., 1.))
    raw_mu_pop = pyro.sample("mu_pop", dist.Normal(0., 1.))
    mu_e_bias = pyro.sample("mu_e_bias", dist.Normal(0., 0.02))
    rho_e_bias = pyro.sample("rho_e_bias", dist.Normal(0.7, 0.1))
    raw_e_bias = pyro.sample("raw_e_bias", dist.Normal(0., 1.))
    raw_measure_noise_national = pyro.sample("measure_noise_national", dist.Normal(0., 1.))
    raw_measure_noise_state = pyro.sample("measure_noise_state", dist.Normal(0., 1.))
    raw_polling_bias = pyro.sample("polling_bias", dist.Normal(0., 1.))

    #Data Transformation
    national_cov_matrix_error_sd = np.sqrt(state_weights.T * state_covariance_0 * state_weights)
    polling_bias = cholesky_ss_cov_poll_bias * raw_polling_bias
    national_polling_bias_average = polling_bias.T * state_weights
    ss_cov_poll_bias = state_covariance_0 * (polling_bias_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_T = state_covariance_0 * (mu_b_T_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_walk = state_covariance_0 * (random_walk_scale/national_cov_matrix_error_sd)**2

    cholesky_ss_cov_poll_bias = torch.cholesky(ss_cov_poll_bias)
    cholesky_ss_cov_mu_b_T = torch.cholesky(ss_cov_mu_b_T)
    cholesky_ss_cov_mu_b_walk = torch.cholesky(ss_cov_mu_b_walk)

    #Model Parameters
    mu_b[:,T] = cholesky_ss_cov_mu_b_T * raw_mu_b_T + mu_b_prior
    for i in range(1,T):
         mu_b[:, T - i] = cholesky_ss_cov_mu_b_walk * raw_mu_b[:, T - i] + mu_b[:, T + 1 - i]
    national_mu_b_average = transpose(mu_b) * state_weights
    mu_c = raw_mu_c * sigma_c
    mu_m = raw_mu_m * sigma_m
    mu_pop = raw_mu_pop * sigma_pop
    e_bias[1] = raw_e_bias[1] * sigma_e_bias
    sigma_rho = sqrt(1-square(rho_e_bias)) * sigma_e_bias
    for t in range(2,T+1):
        e_bias[t] = mu_e_bias + rho_e_bias * (e_bias[t - 1] - mu_e_bias) + raw_e_bias[t] * sigma_rho

    for i in range(1,N_state_polls+1):
        logit_pi_democrat_state[i] = mu_b[state[i], day_state[i]] + mu_c[poll_state[i]] + mu_m[poll_mode_state[i]] + \
            mu_pop[poll_pop_state[i]] + unadjusted_state[i] * e_bias[day_state[i]] + raw_measure_noise_state[i] * sigma_measure_noise_state + \
            polling_bias[state[i]]
  
    logit_pi_democrat_national = national_mu_b_average[day_national] +  mu_c[poll_national] + mu_m[poll_mode_national] + \
        mu_pop[poll_pop_national] + unadjusted_national * e_bias[day_national] + raw_measure_noise_national * sigma_measure_noise_national +\
        national_polling_bias_average; 

    n_democrat_state = torch.Binomial(n_two_share_state, logits = logit_pi_democrat_state)
    n_democrat_national = torch.Binomial(n_two_share_national, logits = logit_pi_democrat_national)