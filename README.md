# Forecasting-Elections-6998
## Presidential Election Model Evaluation

This repository is created by Manik Goyal (mg4106) and Mark Dijkstra (md3507) for [Machine Learning with Probabilistic Programming](http://www.proditus.com/mlpp2020).

## Final Project Notebook
The final project notebook is under `/final-project/final-notebook.ipynb`. This notebook is our final report.
There are some supplemental files alongside in the same folder

- `/final-project/final_2016.py` - This python file is used to clean data and covert it into the required format for the pyro model. This file also runs the OLS(Ordinary Least Square) regression model for "Fundamentals" Abramowitz "Time-For-Change" model.

- `/final-project/poll_model.py` - This file contains the dynamic bayesian model in pyro alongside the functions used for Inference (MCMC; NUTS)

- `/final-project/makepsd.py` - This utility file is used to convert matrices to positive semi-definite matrices

- `/final-project/graph_util.py` - This utility file is used to compute intermediatory arrays for plotting graphs and plot various evaluation graphs.


> the Markdown export of the notebook is  ~ 1500 words (ie within the required word limit of the project)

To compute this, save the Jupyter notebook as a Markdown file by going to
```
File > Download as > Markdown (.md)
```
and then counting the words as follows
```
wc -w final-notebook.md
```

## Development
Use Python 3.8+. (We used Python 3.8.0).

For configuring a virtual environment. Please follow the documentation
[here](https://docs.python.org/3.8/tutorial/venv.html).

Once you activate the virtual environment, use `pip` to install the variety of packages using the follwing command
```{bash}
(venv)$ pip install -r requirements.txt
```

This should install Pyro, along with Jupyter and other useful libraries.

All introduced additional dependencies to the final project, are in the `requirements.txt` with pinned versioning.

### Code styling
Any additional code written by us succesfully passes the `flake8` linting. For more details on falke8 see this
[blog post](https://medium.com/python-pandemonium/what-is-flake8-and-why-we-should-use-it-b89bd78073f2)

To check the flake8 linting do the following after cloning the repository
```{bash}
(venv)$ flake8
```
