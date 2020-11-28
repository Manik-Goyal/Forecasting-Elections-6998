import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS, Predictive
from scipy.special import expit


def convert_to_tensor(dic):
    for k, v in dic.items():
        dic[k] = torch.as_tensor(v)
    return dic


def model(data, polls=None):
    # data from data dictionary

    data = convert_to_tensor(data)
    # X
    N_national_polls = data["N_national_polls"]     # Number of National Polls
    N_state_polls = data["N_state_polls"]           # Number of State Polls
    T = data["T"]    # Number of days
    S = data["S"]    # Number of states for which at-least 1 poll is available
    P = data["P"]    # Number of pollsters
    M = data["M"]    # Number of poll modes
    Pop = data["Pop"]   # Number of poll populations
    state = data["state"]       # State index
    day_state = data["day_state"]   # State Day index
    day_national = data["day_national"]     # National Day index
    poll_state = data["poll_state"]     # State Pollster Index
    poll_national = data["poll_national"]       # National Pollster Index
    poll_mode_state = data["poll_mode_state"]   # State Poll Mode Index
    poll_mode_national = data["poll_mode_national"]  # National Poll Mode Index
    poll_pop_state = data["poll_pop_state"]     # State poll population
    poll_pop_national = data["poll_pop_national"]   # National Poll Populaiton
    unadjusted_national = data["unadjusted_national"]
    unadjusted_state = data["unadjusted_state"]

    # Total Number of Dem+Reb supporters for a particular poll
    n_two_share_national = data["n_two_share_national"]
    # Total Number of Dem+Reb supporters for a particular poll
    n_two_share_state = data["n_two_share_state"]

    # y
    if polls is not None:
        polls = convert_to_tensor(polls)
        # Number of Dem supporters in national poll for a particular poll
        n_democrat_national = polls["n_democrat_national"]
        # Number of Dem supporters in state poll for a particular poll
        n_democrat_state = polls["n_democrat_state"]
    else:
        n_democrat_national = None
        n_democrat_state = None

    # Prior Input values
    mu_b_prior = data["mu_b_prior"]
    state_weights = data["state_weights"]
    sigma_c = data["sigma_c"]
    sigma_m = data["sigma_m"]
    sigma_pop = data["sigma_pop"]
    sigma_measure_noise_national = data["sigma_measure_noise_national"]
    sigma_measure_noise_state = data["sigma_measure_noise_state"]
    sigma_e_bias = data["sigma_e_bias"]

    # Covariance Matrix and Scale Input
    state_covariance_0 = data["state_covariance_0"]
    random_walk_scale = data["random_walk_scale"]
    mu_b_T_scale = data["mu_b_T_scale"]
    polling_bias_scale = data["polling_bias_scale"]

    # Data Transformation
    national_cov_matrix_error_sd = \
        torch.sqrt(state_weights.T @ state_covariance_0 @ state_weights)

    # Scale Covariance
    ss_cov_poll_bias = state_covariance_0 * \
        (polling_bias_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_T = state_covariance_0 * \
        (mu_b_T_scale/national_cov_matrix_error_sd)**2
    ss_cov_mu_b_walk = state_covariance_0 * \
        (random_walk_scale/national_cov_matrix_error_sd)**2

    # Cholesky Transformation
    cholesky_ss_cov_poll_bias = torch.cholesky(ss_cov_poll_bias)
    cholesky_ss_cov_mu_b_T = torch.cholesky(ss_cov_mu_b_T)
    cholesky_ss_cov_mu_b_walk = torch.cholesky(ss_cov_mu_b_walk)

    # Priors
    # Parameters
    with pyro.plate("raw_mu_b_T-plate", size=S):
        raw_mu_b_T = pyro.sample("raw_mu_b_T", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_b_y-plate", size=T):
        with pyro.plate("raw_mu_b_x-asis", size=S):
            raw_mu_b = pyro.sample("raw_mu_b", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_c-plate", size=P):
        raw_mu_c = pyro.sample("raw_mu_c", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_m-plate", size=M):
        raw_mu_m = pyro.sample("raw_mu_m", dist.Normal(0., 1.))

    with pyro.plate("raw_mu_pop-plate", size=Pop):
        raw_mu_pop = pyro.sample("raw_mu_pop", dist.Normal(0., 1.))

    # Not sure if this satisfies Offset=0 and multiplier=0.02
    mu_e_bias = pyro.sample("mu_e_bias", dist.Normal(0., 0.02))*0.02

    rho_e_bias = torch.clamp(pyro.sample("rho_e_bias",
                                         dist.Normal(0.7, 0.1)), 0, 1)

    with pyro.plate("raw_e_bias-plate", size=T):
        raw_e_bias = pyro.sample("raw_e_bias", dist.Normal(0., 1.))

    with pyro.plate("raw_measure_noise_national-plate", size=N_national_polls):
        raw_measure_noise_national = \
            pyro.sample("measure_noise_national", dist.Normal(0., 1.))

    with pyro.plate("raw_measure_noise_state-plate", size=N_state_polls):
        raw_measure_noise_state = \
            pyro.sample("measure_noise_state", dist.Normal(0., 1.))

    with pyro.plate("raw_polling_bias-plate", size=S):
        raw_polling_bias = pyro.sample("raw_polling_bias", dist.Normal(0., 1.))

    # Transformed Parameters
    mu_b = pyro.deterministic('mu_b', torch.empty(S, T))    # Initalize mu_b
    mu_b[:, T-1] = cholesky_ss_cov_mu_b_T @ raw_mu_b_T.double() + mu_b_prior
    for i in range(2, T + 1):
        mu_b[:, T - i] = cholesky_ss_cov_mu_b_walk @ \
            raw_mu_b[:, T - i].double() + mu_b[:, T + 1 - i]

    mu_c = pyro.deterministic('mu_c', raw_mu_c * sigma_c)
    mu_m = pyro.deterministic('mu_m', raw_mu_m * sigma_m)
    mu_pop = pyro.deterministic('mu_pop', raw_mu_pop * sigma_pop)
    sigma_rho = pyro.deterministic('sigma_rho', torch.sqrt(1-(rho_e_bias)**2)
                                   * sigma_e_bias)

    e_bias = pyro.deterministic('e_bias', torch.empty(T))   # Initalize e_bias
    e_bias[0] = raw_e_bias[0] * sigma_e_bias
    for t in range(1, T):
        e_bias[t] = mu_e_bias + rho_e_bias * \
            (e_bias[t - 1] - mu_e_bias) + raw_e_bias[t] * sigma_rho

    polling_bias = pyro.deterministic('polling_bias', cholesky_ss_cov_poll_bias
                                      @ raw_polling_bias.double())

    national_mu_b_average = pyro.deterministic('national_mu_b_average',
                                               mu_b.T.double() @
                                               state_weights.double())
    national_polling_bias_average = \
        pyro.deterministic('national_polling_bias_average',
                           polling_bias.T.double() @
                           state_weights.double())

    logit_pi_democrat_state = pyro.deterministic('logit_pi_democrat_state',
                                                 torch.zeros(N_state_polls))
    logit_pi_democrat_national = \
        pyro.deterministic('logit_pi_democrat_national',
                           torch.zeros(N_national_polls))

    logit_pi_democrat_state = mu_b[state, day_state] + mu_c[poll_state] + \
        mu_m[poll_mode_state] + mu_pop[poll_pop_state] + \
        unadjusted_state * e_bias[day_state] + \
        raw_measure_noise_state * sigma_measure_noise_state + \
        polling_bias[state].float()

    pyro.sample("n_democrat_state",
                dist.Binomial(n_two_share_state,
                              logits=logit_pi_democrat_state),
                obs=n_democrat_state)

    logit_pi_democrat_national = national_mu_b_average[day_national] + \
        mu_c[poll_national] + mu_m[poll_mode_national] + \
        mu_pop[poll_pop_national] + \
        unadjusted_national * e_bias[day_national] + \
        raw_measure_noise_national * sigma_measure_noise_national + \
        national_polling_bias_average

    pyro.sample("n_democrat_national",
                dist.Binomial(n_two_share_national,
                              logits=logit_pi_democrat_national),
                obs=n_democrat_national)


# MCMC with No-U Turn Sampler
def Inference_MCMC(model, data, polls, n_samples=500,
                   n_warmup=500, n_chains=1):

    nuts_kernel = NUTS(model, adapt_step_size=True,
                       jit_compile=True, ignore_jit_warnings=True,
                       max_tree_depth=6)

    mcmc = MCMC(nuts_kernel, num_samples=n_samples, warmup_steps=n_warmup,
                num_chains=n_chains)

    mcmc.run(data, polls)

    # the samples that were not rejected;
    # actual samples from the posterior dist
    posterior_samples = mcmc.get_samples()

    # turning to a dict
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in
                   mcmc.get_samples().items()}

    return posterior_samples, hmc_samples


# Generate samples from posterior predictive distribution
def sample_posterior_predictive(model, posterior_samples, n_samples, data):
    posterior_predictive = Predictive(model, posterior_samples,
                                      num_samples=n_samples)(data, polls=None)

    return posterior_predictive


# Generating Quantity from posterior sample
def predicted_score(posterior_predictive, data):
    T = data['T']
    S = data['S']
    predicted = torch.empty(T, S)
    mu_b = posterior_predictive["mu_b"].squeeze()[-1]

    for s in range(S):
        predicted[:, s] = expit(mu_b[s, :])

    return predicted
