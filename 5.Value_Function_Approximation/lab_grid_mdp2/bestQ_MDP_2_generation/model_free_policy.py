import MDP_2
import random
random.seed(0)

grid = MDP_2.MDP()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()


##############   epsilon greedy policy #####


def epsilon_greedy(qfunc, state, epsilon):
    # max q action
    amax = 0
    key = "%d_%s" % (state, actions[0])
    qmax = qfunc[key]
    for i in xrange(len(actions)):
        key = "%d_%s" % (state, actions[i])
        q = qfunc[key]
        if qmax < q:
            qmax = q
            amax = i

    # probability
    pro = [0.0 for i in xrange(len(actions))]
    pro[amax] += 1 - epsilon
    for i in xrange(len(actions)):
        pro[i] += epsilon / len(actions)

    # choose
    rand = random.random()
    border = 0.0
    for i in xrange(len(actions)):
        border += pro[i]
        if rand < border:
            return actions[i]
    return actions[-1]


################ Different model free RL learning algorithms #####


def MC(num_iter1, epsilon, max_actions=100):
    # J_hist = []
    n = dict()
    qfunc = dict()
    for s in states:
        for a in actions:
            qfunc["%d_%s" % (s, a)] = 0.0
            n["%d_%s" % (s, a)] = 1e-3  # For safety division

    for iter1 in xrange(num_iter1):
        # J_hist.append(compute_error(qfunc))
        s_sample = []
        a_sample = []
        r_sample = []

        s = 1
        t = False
        count = 0
        while t == False:
            if count >= max_actions:
                print 'Counts of continuous actions over %d, MC jump out' % max_actions
                break
            a = epsilon_greedy(qfunc, s, epsilon)
            t, s1, r = grid.transform(s, a)
            s_sample.append(s)
            r_sample.append(r)
            a_sample.append(a)
            s = s1
            count += 1

        g = 0.0
        for i in xrange(len(s_sample) - 1, -1, -1):
            g *= gamma
            g += r_sample[i]

        for i in xrange(len(s_sample)):
            key = "%d_%s" % (s_sample[i], a_sample[i])
            n[key] += 1.0
            qfunc[key] = (qfunc[key] * (n[key] - 1) + g) / n[key]

            g -= r_sample[i]
            g /= gamma

    return qfunc


def SARSA(num_iter1, alpha, epsilon, max_actions=100):
    # J_hist = []
    qfunc = dict()
    for s in states:
        for a in actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0

    for iter1 in xrange(num_iter1):
        # J_hist.append(compute_error(qfunc))
        # Arbitrary Initialization
        s = 1
        a = random.choice(actions)
        t = False
        count = 0
        while t == False:
            if count >= max_actions:
                print 'Counts of continuous actions over %d, SARSA jump out' % max_actions
                break
            key = "%d_%s" % (s, a)
            t, s_next, r = grid.transform(s, a)
            a_next = epsilon_greedy(qfunc, s_next, epsilon)
            key_next = "%d_%s" % (s_next, a_next)
            qfunc[key] += alpha * (r + gamma * qfunc[key_next] - qfunc[key])
            s = s_next
            a = a_next
            count += 1

    return qfunc


def QLearning(num_iter1, alpha, epsilon, max_actions=100):
    # J_hist = []
    qfunc = dict()
    for s in states:
        for a in actions:
            key = "%d_%s" % (s, a)
            qfunc[key] = 0.0

    for iter1 in xrange(num_iter1):
        # J_hist.append(compute_error(qfunc))
        # Arbitrary Initialization
        s = 1
        a = random.choice(actions)
        t = False
        count = 0
        while t == False:
            if count >= max_actions:
                print 'Counts of continuous actions over %d, QLearning jump out' % max_actions
                break
            key = "%d_%s" % (s, a)
            t, s_next, r = grid.transform(s, a)
            # Find next action through the max of Qt(St+1, a)
            qmax = min(qfunc.values())
            key_next = ''
            for a_n in actions:
                key_next0 = "%d_%s" % (s_next, a_n)
                if qmax <= qfunc[key_next0]:
                    qmax = qfunc[key_next0]
                    key_next = key_next0

            qfunc[key] += alpha * (r + gamma * qfunc[key_next] - qfunc[key])
            s = s_next
            a = epsilon_greedy(qfunc, s_next, epsilon)
            count += 1

    return qfunc

if __name__ == "__main__":
    iterations = 10000
    alpha = 0.2
    epsilon = 0.5

    qfunc_Q = QLearning(iterations, alpha, epsilon)
    print 'QLearning', qfunc_Q

    with open('bestQ_MDP_2.txt', 'a') as f:
        for s in states:
            for a in actions:
                f.write('%d_%s:%.5f' %(s, a, qfunc_Q['%d_%s' % (s, a)]))
                f.write('\n')