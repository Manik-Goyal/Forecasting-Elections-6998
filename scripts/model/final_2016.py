import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta

 
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
	cols = ['year', 'state', 'dem']
	df = pd.read_csv("../../data/potus_results_76_16.csv", usecols=cols)
	df = df.loc[df['year'] == 2016]


	return df


def main():
	print("Example usages below for undertanding the code: \n\n")
	print("Using cov_matrix(6, .75, .95): \n", cov_matrix(6, 0.75, 0.95), '\n\n')

	df = read_all_polls()
	print(df)
	df = read_state_context()
	print(df)
	df = create_cov_matrices()
	print(df)


#	state_covariance_polling_bias = cov_matrix(51, 0.078^2, 0.9) # 3.4% on elec day
#	state_covariance_polling_bias <- state_covariance_polling_bias * state_correlation_polling


if __name__ == "__main__":
	main()
