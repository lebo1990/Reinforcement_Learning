import MDP
import random
# random.seed(0)

grid = MDP.MDP()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()


def TD(alpha, gamma, state_sample, action_sample, reward_sample):
    vfunc = dict()
    for s in states:
        vfunc[s] = random.random()

    for iter1 in xrange(len(state_sample)):
        for step in xrange(len(state_sample[iter1])):
            s = state_sample[iter1][step]
            r = reward_sample[iter1][step]

            next_v = 0.0
            if len(state_sample[iter1]) - 1 > step:
                s1 = state_sample[iter1][step + 1]
                next_v = vfunc[s1]

            vfunc[s] = vfunc[s] + alpha * (r + gamma * next_v - vfunc[s])
    return vfunc


if __name__ == "__main__":
    s, a, r = grid.gen_randompi_sample(10000)
    vfunc = TD(0.15, 0.5, s, a, r)
    print "Model-free Temporal-Difference:"
    print vfunc
