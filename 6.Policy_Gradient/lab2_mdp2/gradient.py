from policy_value import *
import random


def update_valuepolicy(valuepolicy, f, a, tvalue, alpha):
    pvalue = valuepolicy.qfunc(f, a)
    error = pvalue - tvalue
    fea = valuepolicy.get_fea_vec(f, a)
    valuepolicy.theta -= alpha * error * fea


def update_softmaxpolicy(softmaxpolicy, f, a, qvalue, alpha):

    fea = softmaxpolicy.get_fea_vec(f, a)
    prob = softmaxpolicy.pi(f)

    delta_logJ = fea
    for i in xrange(len(softmaxpolicy.actions)):
        a1 = softmaxpolicy.actions[i]
        fea1 = softmaxpolicy.get_fea_vec(f, a1)
        delta_logJ -= fea1 * prob[i]
    # delta_logJ *= -1.0

    softmaxpolicy.theta += alpha * delta_logJ * qvalue

################ Different model free RL learning algorithms #####


def MC(grid, softmaxpolicy, num_iter1, alpha, max_count = 100):
    # actions = grid.actions
    gamma = grid.gamma
    for i in xrange(len(policy.theta)):
        policy.theta[i] = 0.1

    for iter1 in xrange(num_iter1):
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
            a = softmaxpolicy.take_action(f)
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
            update_softmaxpolicy(solftmaxpolicy, f_sample[
                                 i], a_sample[i], g, alpha)

            g -= r_sample[i]
            g /= gamma

    return softmaxpolicy


def SARSA(grid, evaler, softmaxpolicy, valuepolicy, num_iter1, alpha, max_count = 100):
    actions = grid.actions
    gamma = grid.gamma
    J_hist = []
    for i in xrange(len(valuepolicy.theta)):
        valuepolicy.theta[i] = 0.0
    for i in xrange(len(softmaxpolicy.theta)):
        softmaxpolicy.theta[i] = 0.0

    for iter1 in xrange(num_iter1):
        J_hist.append(evaler.eval(valuepolicy))
        f = grid.start()
        a = random.choice(actions)
        t = False
        count = 0

        while t == False:
            if count >= max_count:
                print 'Counts of continuous actions over %d, SARSA jump out' % max_count
                break
            t, f1, r = grid.receive(a)
            a1 = softmaxpolicy.take_action(f1)
            update_valuepolicy(valuepolicy, f, a, r + gamma *
                               valuepolicy.qfunc(f1, a1), alpha)
            update_softmaxpolicy(softmaxpolicy, f, a,
                                 valuepolicy.qfunc(f, a), alpha)

            f = f1
            a = a1
            count += 1

    return softmaxpolicy, J_hist


def Qlearning(grid, evaler, softmaxpolicy, valuepolicy, num_iter1, alpha, max_count = 100):
    actions = grid.actions
    gamma = grid.gamma
    J_hist = []
    for i in xrange(len(valuepolicy.theta)):
        valuepolicy.theta[i] = 0.0
    for i in xrange(len(softmaxpolicy.theta)):
        softmaxpolicy.theta[i] = 0.0

    for iter1 in xrange(num_iter1):
        J_hist.append(evaler.eval(valuepolicy))
        f = grid.start()
        a = random.choice(actions)
        t = False
        count = 0

        while t == False:
            if count >= max_count:
                print 'Counts of continuous actions over %d, Q-learning jump out' % max_count
                break
            t, f1, r = grid.receive(a)
            # Initilization
            qmax = 0
            for ai in actions:
                qmax = min(qmax, valuepolicy.qfunc(f1, ai))
            a1 = ''
            # Find next action through the max of Qt(St+1, a)
            for a_n in actions:
                q1 = valuepolicy.qfunc(f1, a_n)
                if qmax < q1:
                    qmax = q1
                    a1 = a_n

            # Random Update
            if a1 == '':
                a1 = random.choice(actions)

            update_valuepolicy(valuepolicy, f, a, r + gamma *
                               valuepolicy.qfunc(f1, a1), alpha)
            update_softmaxpolicy(softmaxpolicy, f, a,
                                 valuepolicy.qfunc(f, a), alpha)

            f = f1
            a = a1
            count += 1

    return softmaxpolicy, J_hist