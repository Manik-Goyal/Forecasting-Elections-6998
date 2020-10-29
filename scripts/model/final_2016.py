import numpy as np
import pandas as pd


def cov_matrix(n, sigma2, rho):
	m = np.ones(shape=(n,n)) * rho
	m2 = np.zeros(shape=(n,n))

	np.fill_diagonal(m, 1)
	np.fill_diagonal(m2, sigma2 ** .5)
	
	return(np.matmul(np.matmul(m2, m), m2))

'''
def mean_low(draws, states, identity):
	temp = draws
	draws_df 
'''


#def check_cov_matrix(matrix, wt=state_weights):




def main():
	print("Example usages below for undertanding the code: \n\n")
	print("Using cov_matrix(6, .75, .95): \n", cov_matrix(6, 0.75, 0.95), '\n\n')

	df = pd.read_csv("../../data/all_polls.csv")
	df = df.drop(df.columns[0] , axis=1)
	print(df)


if __name__ == "__main__":
	main()
