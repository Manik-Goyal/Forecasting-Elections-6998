from collections import defaultdict


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
