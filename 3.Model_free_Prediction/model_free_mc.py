import MDP

grid = MDP.MDP()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()


def MC(gamma, state_sample, action_sample, reward_sample):
    vfunc = dict()
    nfunc = dict()
    for s in states:
        vfunc[s] = 0.0
        nfunc[s] = 0.0

    for iter1 in xrange(len(state_sample)):
        G = 0.0
        for step in xrange(len(state_sample[iter1]) - 1, -1, -1):
            G *= gamma
            G += reward_sample[iter1][step]
        for step in xrange(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            vfunc[s] += G
            nfunc[s] += 1.0
            G -= reward_sample[iter1][step]
            G /= gamma

    for s in states:
        if nfunc[s] > 0:
            vfunc[s] /= nfunc[s]
        else:
            vfunc[s] = 0
    return vfunc


if __name__ == "__main__":
    s, a, r = grid.gen_randompi_sample(10000)
    vfunc = MC(0.5, s, a, r)
    print "Model-free Monte Carlo:"
    print vfunc
