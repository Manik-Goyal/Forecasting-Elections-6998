from collections import defaultdict
from scipy.special import expit
import numpy as np
import pandas as pd
import torch
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


def average(vals):
    return sum(vals) / len(vals)


def std(vals, mu):
    var = sum([((x - mu) ** 2) for x in vals]) / len(vals)
    return var ** .5


# This function creates the dictionary of predictions
# for a specific state given an array of state predictions
# for a given day with an arbitrary number of simulations
# to determine that day
def createPredictionsDict(vals):
    preds = defaultdict(list)
    for i in range(len(vals)):
        temp = vals[i]
        for j in range(len(temp)):
            preds[j].append(temp[j])
    return preds


# This calculates various details needed for plotting
# including number of wins for Trump vs Clinton
# and the electoral vote splits
def electoralVoteCalculator(numDays, numStates, preds, EV_Index, EV):
    EV_list_trump = list()
    EV_list_clinton = list()
    clintonWins = 0
    trumpWins = 0
    for i in range(numDays):
        evTotal_clinton = 0
        evTotal_trump = 0
        for j in range(numStates):
            pct = preds[j][i]
            if pct >= 0.5:
                evTotal_clinton += EV[EV_Index[j]]
            else:
                evTotal_trump += EV[EV_Index[j]]
        evTotal_clinton = int(evTotal_clinton)
        evTotal_trump = int(evTotal_trump)
        if evTotal_clinton > evTotal_trump:
            clintonWins += 1
        else:
            trumpWins += 1
        EV_list_clinton.append(evTotal_clinton)
        EV_list_trump.append(evTotal_trump)
    return clintonWins, trumpWins, EV_list_trump, EV_list_clinton


def mean_low_high(draws, states, Id):
    mean = expit(draws.mean(axis=0))
    high = expit(draws.mean(axis=0) + 1.96 * draws.std(axis=0))
    low = expit(draws.mean(axis=0) - 1.96 * draws.std(axis=0))
    Id = [Id] * len(states)
    draws_df = {'states': states, 'mean': mean, 'high': high,
                'low': low, 'type': Id}
    draws_df = pd.DataFrame(draws_df)

    return draws_df


def function_tibble(x, predicted):
    temp = predicted[:, :, x]
    low = np.quantile(predicted[:, :, x], 0.025, axis=0)
    high = np.quantile(predicted[:, :, x], 0.975, axis=0)
    mean = torch.mean(predicted[:, :, x], axis=0)
    prob = (predicted[:, :, x] > 0.5).type(torch.float).mean(axis=0)
    state = [x] * temp.shape[1]
    t = np.arange(temp.shape[1])
    df = pd.DataFrame({'low': low, 'high': high, 'mean': mean,
                       'prob': prob, 'state': state, 't': t})

    return df


def generate_mu_sigma(EV_C_mu, EV_C_std, EV_T_mu, EV_T_std,
                      cWinsList, tWinsList, predicted,
                      numDays, numStates, EV_Index, EV):
    for i in tqdm(range(252)):
        vals = predicted[:, i, :]
        pred = createPredictionsDict(vals)
        out = electoralVoteCalculator(numDays, numStates,
                                      pred, EV_Index, EV)
        cWins, tWins, EV_T, EV_C = out[0], out[1], out[2], out[3]
        cWinsList.append(cWins)
        tWinsList.append(tWins)

        mu_C, mu_T = average(EV_C), average(EV_T)
        std_C, std_T = std(EV_C, mu_C), std(EV_T, mu_T)

        EV_C_mu.append(mu_C)
        EV_C_std.append(std_C)

        EV_T_mu.append(mu_T)
        EV_T_std.append(std_T)

    return EV_C_mu, EV_C_std, EV_T_mu, EV_T_std, \
        cWinsList, tWinsList


def plot_ev_over_time(EV_C_mu, EV_C_std, EV_T_mu, EV_T_std):
    upper_C = [x + y for x, y in zip(EV_C_mu, EV_C_std)]
    lower_C = [x - y for x, y in zip(EV_C_mu, EV_C_std)]

    upper_T = [x + y for x, y in zip(EV_T_mu, EV_T_std)]
    lower_T = [x - y for x, y in zip(EV_T_mu, EV_T_std)]

    upper_2C = [x + 2*y for x, y in zip(EV_C_mu, EV_C_std)]
    lower_2C = [x - 2*y for x, y in zip(EV_C_mu, EV_C_std)]

    upper_2T = [x + 2 * y for x, y in zip(EV_T_mu, EV_T_std)]
    lower_2T = [x - 2 * y for x, y in zip(EV_T_mu, EV_T_std)]

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    sns.lineplot(data=np.array(EV_C_mu), color="blue")
    sns.lineplot(data=np.array(EV_T_mu), color="red")
    plt.fill_between(range(len(EV_C_mu)), upper_C, lower_C,
                     facecolor="blue", alpha=0.25)
    plt.fill_between(range(len(EV_C_mu)), upper_2C, lower_2C,
                     facecolor="blue", alpha=0.15)

    plt.fill_between(range(len(EV_T_mu)), upper_T, lower_T,
                     facecolor="red", alpha=0.25)
    plt.fill_between(range(len(EV_T_mu)), upper_2T, lower_2T,
                     facecolor="red", alpha=0.15)

    plt.annotate(' Clinton: %d' % EV_C_mu[-1],
                 xy=(252, EV_C_mu[-1]), fontsize=12, color='blue')
    plt.annotate(' Trump: %d' % EV_T_mu[-1],
                 xy=(252, EV_T_mu[-1]), fontsize=12, color='red')

    plt.axhline(y=270, color='black', linestyle='-', xmax=0.90)
    plt.text(50, 270, '270 to win', fontsize=12, va='center',
             ha='center', backgroundcolor='w')
    plt.axvline(x=252, color='black', linestyle='-')
    plt.text(252, 400, 'Election Day', fontsize=12, va='center',
             ha='center', backgroundcolor='w')
    plt.legend(['Clinton', 'Trump'], bbox_to_anchor=(0.5, -0.15))
    plt.xlabel("Day")
    plt.ylabel("Electoral Votes")
    plt.title("Number of Electoral Votes over Time w. Deviation")
    plt.xlim(0, 280)
    plt.show()
    return


def summary(v):
    site_stats = {
            "mean": torch.mean(v, 0),
            "std": torch.std(v, 0),
            "5%": v.kthvalue(int(len(v) * 0.05), dim=0)[0],
            "95%": v.kthvalue(int(len(v) * 0.95), dim=0)[0],
        }

    return site_stats


def plot_posterior_predictive_check(obs, y_true, n_polls):
    x = sorted(np.random.randint(0, n_polls, 500))
    y = summary(obs)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 6), sharey=True)
    fig.suptitle("Posterior Predictive Check", fontsize=16)
    ax[0].plot(x, np.take(y_true, x), "o")
    ax[0].plot(x, np.take(y['mean'], x))
    ax[0].set_xlabel("Predicted mean vs True", fontsize=15)

    ax[1].plot(x, np.take(y_true, x), "o")
    ax[1].fill_between(x, np.take(y["5%"], x),
                       np.take(y["95%"], x),
                       alpha=0.8)
    ax[1].set_xlabel("Predicted with 90% CI vs True", fontsize=15)

    return
