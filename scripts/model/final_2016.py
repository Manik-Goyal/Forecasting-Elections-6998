import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import timedelta
import copy
from scipy.special import logit

 
NaN = float('nan')


def cov_matrix(n, sigma2, rho):
	m = np.ones(shape=(n,n)) * rho
	m2 = np.zeros(shape=(n,n))

	np.fill_diagonal(m, 1)
	np.fill_diagonal(m2, sigma2 ** .5)
	
	return(np.matmul(np.matmul(m2, m), m2))

def check_cov_matrix(mat, weights):
	diag = np.diag(mat)



def fit_rmse_day_x(x):
	y = []
	for num in x:
		y.append( 0.03 + (10 **(-6.6)) * (num ** 2) )
	return y	

def pass_data():
	cols = ['state', 'pollster', 'number.of.observations','population', 'mode',
		'start.date',
		'end.date',
		'clinton', 'trump', 'undecided', 'other', 'johnson', 'mcmullin']
	df = pd.read_csv("../../data/all_polls.csv", usecols=cols)
	#       pd.set_option("display.max_rows", None, "display.max_columns", None)
	#a + (b - a)/2
	# Mutations
	df = df.rename(columns={"number.of.observations": "n", "population": "polltype"})
	df['begin'] = [datetime.strptime(x, "%Y-%m-%d") for x in df['start.date']]
	df['end'] = [datetime.strptime(x, "%Y-%m-%d") for x in df['end.date']]
	df['t'] = [(y - (timedelta(days=1) + (y - x) // 2) ).date() for x,y in zip(df['begin'], df['end'])]

	# Matching equivalent names due to data inconsistencies
	df = df.replace("Fox News", "FOX")
	df = df.replace("WashPost", "Washington Post")
	df = df.replace("ABC News", "ABC")
	df = df.replace("DHM Research", "DHM")
	df = df.replace("Public Opinion Strategies", "POS")
	df['undecided'] = df['undecided'].replace(NaN, 0)
	df['other'] = df['other'].replace(NaN, 0) + df['johnson'].replace(NaN, 0) + df['mcmullin'].replace(NaN, 0)

	# Vote shares etc
	df['two_party_sum'] = df['clinton'] + df['trump']
	df['n_clinton'] = round(df['n'] * df['clinton'] / 100)
	df['pct_clinton'] = df['clinton'] / df['two_party_sum']
	df['n_trump'] = round(df['n'] * df['trump'] / 100)
	df['pct_trump'] = df['trump'] / df['two_party_sum']

	# Numerical Indices
	df['index_s'] = pd.Categorical(df['state']).codes
	df['index_s'] = df['index_s'].replace(0, 52)

	min_T = df['t'].min()
	df['index_t'] = [(timedelta(days=1)  + x - min_T).days for x in df['t']]
	df['index_p'] = pd.Categorical(df['pollster']).codes
	df['index_m'] = pd.Categorical(df['mode']).codes
	df['index_pop'] = pd.Categorical(df['polltype']).codes


	####################################################################



	dfT = pd.read_csv("../../data/2012.csv")

	national_score = sum( dfT['obama_count'] ) / sum( dfT['obama_count'] + dfT['romney_count'] )

	dfT['score'] = dfT['obama_count'] / (dfT['romney_count'] + dfT['obama_count'])
	dfT['national_score'] = [national_score] * len( dfT['score'] )
	dfT['delta'] = dfT['score'] - dfT['national_score']
	dfT['share_national_vote'] = dfT['total_count']*(1+dfT['adult_pop_growth_2011_15']) /sum(dfT['total_count']*(1+dfT['adult_pop_growth_2011_15']))

	# set prior differences
	sum_share_national_vote = sum(dfT['share_national_vote'])
	col_names = dfT['state'].tolist()
	prior_diff_score = dfT['delta']
	prior_diff_score.index = col_names
	#       prior_diff_score = {key: value for (key, value) in zip(df['state'], df['delta'])}

	# set state weights
	sum_share_national_vote = sum(dfT['share_national_vote'])
	col_names = dfT['state'].tolist()
	state_weights = dfT['ev'] / sum_share_national_vote
	state_weights.index = col_names

	# set electoral votes per state
	ev_state = {key: value for (key, value) in zip(dfT['state'], dfT['ev'])}

	#######################################################################

	cols = ['state', 'year', 'dem']
	temp = pd.read_csv("../../data/potus_results_76_16.csv", usecols=cols)
	states = temp.loc[temp['year'] == 2016].copy()
	states = states.reset_index()
	states = states.dropna()
	states = states.rename(columns={'year': 'variable', 'dem':'value'})
	states = states.drop(columns=['index'])
	#       print("States: \n\n ", states, "\n\n")

	cols = ['white_pct', 'black_pct', 'hisp_other_pct', 'college_pct', 'wwc_pct', 'median_age', 'state']
	census = pd.read_csv('../../data/acs_2013_variables.csv', usecols=cols)
	census = census.dropna()
	census_size = len(census['state'])
	for col in cols[:-1]:
		dfTemp = census[['state', col]]
		dfTemp = dfTemp.rename(columns={col: 'value'})
		dfTemp.insert(1, 'variable', [col] * census_size)
		states = states.append(dfTemp, ignore_index=True)

	cols = ['state', 'average_log_pop_within_5_miles']
	urban = pd.read_csv('../../data/urbanicity_index.csv', usecols=cols)
	urban = urban.dropna()
	urban = urban.rename(columns={'average_log_pop_within_5_miles':'value'})
	urban.insert(1, 'variable', ['pop_density'] * len(urban['state']))
	states = states.append(urban, ignore_index=True)


	white_evangel = pd.read_csv('../../data/white_evangel_pct.csv')
	white_evangel = white_evangel.dropna()
	white_evangel = white_evangel.rename(columns={'pct_white_evangel':'value'})
	white_evangel.insert(1, 'variable', ['pct_white_evangel'] * len(white_evangel['state']))
	states = states.append(white_evangel, ignore_index=True)

	state_abbrv = states.state.unique()
	final_df = pd.DataFrame(columns = state_abbrv)
	states = states.pivot(index='variable', columns='state', values='value')
	states.fillna(0)

	indices = list(states.index.values)
	mins, maxes = states.min(axis=1).tolist(), states.max(axis=1).tolist()
	print(indices, mins)
	for i in range(len(indices)):
		states.iloc[i] = (states.iloc[i] - mins[i]) / (maxes[i] - mins[i])


	# Our covariance matrix for the data we have transformed        
	cov = states.cov()
	cov[cov < 0 ] = 0


	temp_cov = copy.deepcopy(cov)
	np.fill_diagonal(temp_cov.values, float('NaN'))


	lamda = 0.75
	arr = pd.DataFrame(data=np.ones((51,51)))
	a = 1

	state_correlation_polling = lamda * cov + (1 - lamda) * arr
	df1 = lamda * cov
	df2 = (1 - lamda) * arr
	df2.columns = df1.columns
	df2.index = df1.index
	state_correlation_polling = df1.add(df2)


	# covariance matrix for polling error
	state_covariance_polling_bias =  pd.DataFrame(data=cov_matrix(51, 0.078**2, 0.9), columns = df1.columns, index=df1.index)
	state_covariance_polling_bias = state_covariance_polling_bias * state_correlation_polling

	# covariance for prior e-day prediction
	state_covariance_mu_b_T = pd.DataFrame(data=cov_matrix(51, 0.18**2, 0.9), columns = df1.columns, index=df1.index)
	state_covariance_mu_b_T = state_covariance_mu_b_T * state_correlation_polling

	# covariance matrix for random walks
	state_covariance_mu_b_walk = pd.DataFrame(cov_matrix(51, 0.017**2, 0.9), columns = df1.columns, index=df1.index)
	state_covariance_mu_b_walk = state_covariance_mu_b_walk * state_correlation_polling # we want the demo correlations for filling in gaps in the polls


	## MAKE DEFAULT COV MATRICES
	# we're going to make TWO covariance matrix here and pass it
	# and 3 scaling values to stan, where the 3 values are 
	# (1) the national sd on the polls, (2) the national sd
	# on the prior and (3) the national sd of the random walk
	# make initial covariance matrix (using specified correlation)
	state_covariance_0 = pd.DataFrame(cov_matrix(51, 0.07**2, 0.9), columns = df1.columns, index=df1.index)
	state_covariance_0 = state_covariance_0 * state_correlation_polling # we want the demo correlations for filling in gaps in the polls

	# save the inital scaling factor
	national_cov_matrix_error_sd = ( (state_weights.transpose().dot(state_covariance_0)).dot( state_weights ) ) **0.5 # @TODO

	days_til_election = [100] # as.numeric(difftime(election_day,RUN_DATE))
	expected_national_mu_b_T_error = fit_rmse_day_x(days_til_election)[0]

	polling_bias_scale = 0.013 # on the probability scale -- we convert later down
	mu_b_T_scale = expected_national_mu_b_T_error # on the probability scale -- we convert later down
	random_walk_scale =  0.05/(300 ** 0.5) # on the probability scale -- we convert later down

	# gen fake matrices, check the math (this is recreated in stan
	ss_cov_poll_bias = state_covariance_0 * (polling_bias_scale/national_cov_matrix_error_sd*4) ** 2
	ss_cov_mu_b_T = state_covariance_0 * (mu_b_T_scale/national_cov_matrix_error_sd*4) ** 2
	ss_cov_mu_b_walk = state_covariance_0 * (random_walk_scale/national_cov_matrix_error_sd*4) ** 2

	check_cov_matrix(ss_cov_poll_bias, state_weights)


	cols = ['incvote', 'juneapp', 'q2gdp']
	dfTemp = pd.read_csv("../../data/abramowitz_data.csv", usecols=cols)

	model = smf.ols(formula='incvote ~  juneapp + q2gdp', data = dfTemp) 
	res = model.fit()
	print(res.summary())
	national_mu_prior = 49.6070 + .1393 * 4 + .4480 * 1.1 # use the OLS to do this manually somehow @TODO
	#       ynewpred =  model.predict(df) # predict out of sample

	# on correct scale
	national_mu_prior = national_mu_prior / 100 
	# Mean of the mu_b_prior
	mu_b_prior = logit(national_mu_prior + prior_diff_score)



	#@TODO need to fix the df; it's too big for some reason 
	N_state_polls = len(df.loc[df['index_s'] != 52 ].index)
	N_national_polls = len(df.loc[df['index_s'] == 52].index)

	S = 51
	P = len(df['pollster'].unique())
	M = len(df['mode'].unique()) #@TODO switch to 'method' and do mutation
	Pop = len(df['polltype'].unique())

	state = df.loc[df['index_s'] != 52]['index_s'].tolist() 
	day_national = df.loc[df['index_s'] == 52]['index_t'].tolist() 
	day_state = df.loc[df['index_s'] != 52]['index_t'].tolist()
	poll_national = df.loc[df['index_s'] == 52]['index_p'].tolist()
	poll_state = df.loc[df['index_s'] != 52]['index_p'].tolist()
	poll_mode_national = df.loc[df['index_s'] != 52]['index_m'].tolist()
	poll_mode_state = df.loc[df['index_s'] == 52]['index_m'].tolist()
	poll_pop_national = df.loc[df['index_s'] == 52]['index_pop'].tolist()
	poll_pop_state = df.loc[df['index_s'] == 52]['index_pop'].tolist()


	n_democrat_national = df.loc[df['index_s'] == 52]['n_clinton'].tolist()
	n_democrat_state = df.loc[df['index_s'] != 52]['n_clinton'].tolist()

	n_two_share_national = df.loc[df['index_s'] == 52]['n_trump'].tolist() + n_democrat_national
	n_two_share_state = df.loc[df['index_s'] != 52]['n_trump'].tolist() + n_democrat_state

	#T <- as.integer(round(difftime(election_day, first_day)))
	#current_T <- max(df$poll_day)

	#unadjusted_national <- df %>% mutate(unadjusted = ifelse(!(pollster %in% adjusters), 1, 0)) %>% filter(index_s == 52) %>% pull(unadjusted)
	#unadjusted_state <- df %>% mutate(unadjusted = ifelse(!(pollster %in% adjusters), 1, 0)) %>% filter(index_s != 52) %>% pull(unadjusted)

	# priors (on the logit scale)
	sigma_measure_noise_national = 0.04
	sigma_measure_noise_state = 0.04
	sigma_c = 0.06
	sigma_m = 0.04
	sigma_pop = 0.04
	sigma_e_bias = 0.02


	# putting in a dictionary to give to the pyro model
	data = {}
	data['N_national_polls'] = N_national_polls
	data["N_state_polls"] = N_state_polls
#	data["T"] = T
	data["S"] = S
	data["P"] = P
	data["M"] = M
	data["Pop"] = Pop
	data["state"] = state
	data["state_weights"] = state_weights
	data["day_state"] = day_state
	data["day_national"] = day_national
	data["poll_state"] = poll_state
	data["poll_national"] = poll_national
	data["poll_mode_national"] = poll_mode_national 
	data["poll_mode_state"] = poll_mode_state
	data["poll_pop_national"] = poll_pop_national
	data["poll_pop_state"] = poll_pop_state
#	data["unadjusted_national"] = unadjusted_national
#	data["unadjusted_state"] = unadjusted_state
	data["n_democrat_national"] = n_democrat_national
	data["n_democrat_state"] = n_democrat_state
	data["n_two_share_national"] = n_two_share_national
	data["n_two_share_state"] = n_two_share_state
	data["sigma_measure_noise_national"] = sigma_measure_noise_national
	data["sigma_measure_noise_state"] = sigma_measure_noise_state
	data["mu_b_prior"] = mu_b_prior
	data["sigma_c"] = sigma_c
	data["sigma_m"] = sigma_m
	data["sigma_pop"] = sigma_pop
	data["sigma_e_bias"] = sigma_e_bias
	# covariance matrices
	# ss_cov_mu_b_walk = state_covariance_mu_b_walk,
	# ss_cov_mu_b_T = state_covariance_mu_b_T,
	# ss_cov_poll_bias = state_covariance_polling_bias
	data["state_covariance_0"] = state_covariance_0
	data["polling_bias_scale"] = polling_bias_scale
	data["mu_b_T_scale"] = mu_b_T_scale
	data["random_walk_scale"] = random_walk_scale
	return data


def main():
	pd.set_option("display.max_rows", None, "display.max_columns", None)
	print("Example usages below for undertanding the code: \n\n")
	print("Using cov_matrix(6, .75, .95): \n", cov_matrix(6, 0.75, 0.95), '\n\n')
	data = pass_data()

#	print(fit_rmse_day_x(x))i
#	state_covariance_polling_bias = cov_matrix(51, 0.078^2, 0.9) # 3.4% on elec day
#	state_covariance_polling_bias <- state_covariance_polling_bias * state_correlation_polling


if __name__ == "__main__":
	main()

