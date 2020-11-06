import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
import copy

 
NaN = float('nan')
def cov_matrix(n, sigma2, rho):
	m = np.ones(shape=(n,n)) * rho
	m2 = np.zeros(shape=(n,n))

	np.fill_diagonal(m, 1)
	np.fill_diagonal(m2, sigma2 ** .5)
	
	return(np.matmul(np.matmul(m2, m), m2))



def read_all_polls():
	cols = ['state', 'pollster', 'number.of.observations','population', 'mode', 
		'start.date', 
		'end.date',
		'clinton', 'trump', 'undecided', 'other', 'johnson', 'mcmullin']
	df = pd.read_csv("../../data/all_polls.csv", usecols=cols)
#	pd.set_option("display.max_rows", None, "display.max_columns", None)
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
#	df['index_p'] = 

	return df

#def read_potus_results_76_16(df):


def read_state_context():
	df = pd.read_csv("../../data/2012.csv")	

	national_score = sum( df['obama_count'] ) / sum( df['obama_count'] + df['romney_count'] )

	df['score'] = df['obama_count'] / (df['romney_count'] + df['obama_count'])
	df['national_score'] = [national_score] * len( df['score'] )
	df['delta'] = df['score'] - df['national_score']
	df['share_national_vote'] = df['total_count']*(1+df['adult_pop_growth_2011_15']) /sum(df['total_count']*(1+df['adult_pop_growth_2011_15']))

	# set prior differences
	prior_diff_score = {key: value for (key, value) in zip(df['state'], df['delta'])}
	
	# set state weights
	sum_share_national_vote = sum(df['share_national_vote'])
	state_weights = {key: value / sum_share_national_vote for (key, value) in zip(df['state'], df['share_national_vote']) }

	# set electoral votes per state
	ev_state = {key: value for (key, value) in zip(df['state'], df['ev'])}

	return df

def create_cov_matrices():
	cols = ['state', 'year', 'dem']
	df = pd.read_csv("../../data/potus_results_76_16.csv", usecols=cols)
	states = df.loc[df['year'] == 2016].copy()
	states = states.reset_index()
	states = states.dropna()
	states = states.rename(columns={'year': 'variable', 'dem':'value'})
	states = states.drop(columns=['index'])
#	print("States: \n\n ", states, "\n\n")

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









	return states


def main():
	pd.set_option("display.max_rows", None, "display.max_columns", None)
	print("Example usages below for undertanding the code: \n\n")
	print("Using cov_matrix(6, .75, .95): \n", cov_matrix(6, 0.75, 0.95), '\n\n')

	df = read_all_polls()
	print(df)
	df = read_state_context()
	print(df)
	df = create_cov_matrices()


#	state_covariance_polling_bias = cov_matrix(51, 0.078^2, 0.9) # 3.4% on elec day
#	state_covariance_polling_bias <- state_covariance_polling_bias * state_correlation_polling


if __name__ == "__main__":
	main()

