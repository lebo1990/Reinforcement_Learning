# import grid_mdp
# import evaluate
import random
# import numpy as np


def update(policy, f, a, tvalue, alpha):
    pvalue = policy.qfunc(f, a)
    error = pvalue - tvalue
    fea = policy.get_fea_vec(f, a)
    policy.theta -= alpha * error * fea

################ Different model free RL learning algorithms #####


def MC(grid, policy, evaler, num_iter1, alpha, max_count=100):
    # actions = grid.actions
    gamma = grid.gamma
    J_hist = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.0

    for iter1 in xrange(num_iter1):
        J_hist.append(evaler.eval(policy))
        s_sample = []
        f_sample = []
        a_sample = []
        r_sample = []

        f = grid.start()
        t = False
        count = 0
        while t == False:
            if count >= max_count:
                print 'Counts of continuous actions over %d, MC jump out' % max_count
                break
            a = policy.epsilon_greedy(f)
            s_sample.append(grid.current)
            t, f1, r = grid.receive(a)
            f_sample.append(f)
            r_sample.append(r)
            a_sample.append(a)
            f = f1
            count += 1

        g = 0.0
        for i in xrange(len(f_sample) - 1, -1, -1):
            g *= gamma
            g += r_sample[i]

        for i in xrange(len(f_sample)):
            update(policy, f_sample[i], a_sample[i], g, alpha)

            g -= r_sample[i]
            g /= gamma

    return policy, J_hist


def SARSA(grid, policy, evaler, num_iter1, alpha, max_count=100):
    actions = grid.actions
    gamma = grid.gamma
    J_hist = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        J_hist.append(evaler.eval(policy))
        f = grid.start()
        a = random.choice(actions)
        t = False
        count = 0

        while t == False:
            if count >= max_count:
                print 'Counts of continuous actions over %d, SARSA jump out' % max_count
                break
            t, f1, r = grid.receive(a)
            a1 = policy.epsilon_greedy(f1)
            update(policy, f, a, r + gamma * policy.qfunc(f1, a1), alpha)

            f = f1
            a = a1
            count += 1

    return policy, J_hist


def Qlearning(grid, policy, evaler, num_iter1, alpha, max_count=100):
    actions = grid.actions
    gamma = grid.gamma
    J_hist = []
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
        J_hist.append(evaler.eval(policy))

        f = grid.start()
        a = random.choice(actions)
        t = False
        count = 0

        while t == False:
            if count >= max_count:
                print 'Counts of continuous actions over %d, Qlearning jump out' % max_count
                break
            t, f1, r = grid.receive(a)
            qmax = -1.0
            for a1 in actions:
                pvalue = policy.qfunc(f1, a1)
                if qmax < pvalue:
                    qmax = pvalue
            update(policy, f, a, r + gamma * qmax, alpha)

            f = f1
            a = policy.epsilon_greedy(f)
            count += 1

    return policy, J_hist
