import MDP_2
import random
import numpy as np
#random.seed(0)

grid = MDP_2.MDP()
states = grid.getStates()
actions = grid.getActions()
gamma = grid.getGamma()


##############   epsilon greedy policy #####
# epsilon(s) = n(s) ^ -0.5
# n(s) is the number of times state s has been visited

def epsilon_greedy(qfunc, n_state, state):
    # epsilon Calculation
    epsilon = (n_state[state] + 1) ** -0.5
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


################ Q_learning and Double Q_learning algorithms #####

def QLearning(num_iterations, alpha = [0, 1]):
    """
    Q-learning

    Inputs:  
    - num_iterations: No. of iterations
    - alpha: Fixed Learning_Rate or heuristic Learning_Rate
        if alpha[0] != 0
            Learning_Rate = alpha[0]
        else:
            Learning_Rate = 1 / N(s, a) ^ -alpha[1]
    
    Outputs:
    - max_qfunc_hist: maximum of the Q for every iteration
    - reward_hist: Reward for every iteration
    - track: accumulated N(s, a)

    """
    max_qfunc_hist = []
    reward_hist = []
    state_hist = {}
    qfunc = {}
    track = {}
    
    for si in states:
        for ai in actions:
            key = "%d_%s" % (si, ai)
            qfunc[key] = 0.0

    for iter1 in xrange(num_iterations):
        # Initialization
        for si in states:
            state_hist[si] = 0
        s = 1
        a = random.choice(actions)
        t = False
        r_hist = []
        
        for si in states:
            for ai in actions: 
                key = "%d_%s" % (si, ai)
                track[key] = 0.0

        while t == False:
            key = "%d_%s" % (s, a)
            t, s_next, r = grid.transform(s, a)
            r_hist.append(r)
            # Find next action through the max of Qt(St+1, a)
            qmax = min(qfunc.values())
            key_next = ''
            for a_n in actions:
                key_next0 = "%d_%s" % (s_next, a_n)
                if qmax <= qfunc[key_next0]:
                    qmax = qfunc[key_next0]
                    key_next = key_next0

            track[str(s) + '_' + a] += 1
            if alpha[0] != 0:
                LR = alpha[0]
            else:
                LR = track[str(s) + '_' + a] ** -alpha[1]
            qfunc[key] += LR * (r + gamma * qfunc[key_next] - qfunc[key])
            a = epsilon_greedy(qfunc, state_hist, s)
            state_hist[s] += 1
            s = s_next
            
        reward_hist.append(np.mean(r_hist))
        q1_a = []
        for ai in actions:
            key = "1_%s" % ai
            q1_a.append(qfunc[key])

        max_qfunc_hist.append(max(q1_a))

    return max_qfunc_hist, reward_hist, track


def Double_QLearning(num_iterations, alpha = [0, 1]):
    """
    Double Q-learning

    Inputs:  
    - num_iterations: No. of iterations
    - alpha: Fixed Learning_Rate or heuristic Learning_Rate
        if alpha[0] != 0
            Learning_Rate = alpha[0]
        else:
            Learning_Rate = 1 / N(s, a) ^ -alpha[1]
    
    Outputs:
    - max_qfunc_hist: maximum of the Q for every iteration
    - reward_hist: Reward for every iteration
    - track: accumulated N(s, a)

    """
    max_qfunc_hist = []
    reward_hist = []
    state_hist = {}
    qfunc_a = {}
    qfunc_b = {}
    track = {}
    
    for si in states:
        for ai in actions:
            key = "%d_%s" % (si, ai)
            qfunc_a[key] = 0.0
            qfunc_b[key] = 0.0

    for iter1 in xrange(num_iterations):
        # Initialization
        for si in states:
            state_hist[si] = 0
        s = 1
        a = random.choice(actions)
        t = False
        r_hist = []
        
        for si in states:
            for ai in actions: 
                key = "%d_%s" % (si, ai)
                track[key] = 0.0

        while t == False:
            key = "%d_%s" % (s, a)
            t, s_next, r = grid.transform(s, a)
            r_hist.append(r)
            # Find next action through the max of Qt(St+1, a)
            qmax = min(qfunc_a.values() + qfunc_b.values())
            key_next = ''            

            track[str(s) + '_' + a] += 1
            if alpha[0] != 0:
                LR = alpha[0]
            else:
                LR = track[str(s) + '_' + a] ** -alpha[1]

            Q_choice = random.random()
            # Update A
            if Q_choice < 0.5:
                for a_n in actions:
                    key_next0 = "%d_%s" % (s_next, a_n)
                    if qmax <= qfunc_a[key_next0]:
                        qmax = qfunc_a[key_next0]
                        key_next = key_next0
                qfunc_a[key] += LR * \
                    (r + gamma * qfunc_b[key_next] - qfunc_a[key])
                a = epsilon_greedy(qfunc_a, state_hist, s)
            # Update B
            else:
                for a_n in actions:
                    key_next0 = "%d_%s" % (s_next, a_n)
                    if qmax <= qfunc_b[key_next0]:
                        qmax = qfunc_b[key_next0]
                        key_next = key_next0
                qfunc_b[key] += LR * \
                    (r + gamma * qfunc_a[key_next] - qfunc_b[key])
                a = epsilon_greedy(qfunc_b, state_hist, s)
            
            state_hist[s] += 1
            s = s_next

        reward_hist.append(np.mean(r_hist))
        q1_a = []
        for ai in actions:
            key = "1_%s" % ai
            q1_a.append(qfunc_a[key])
            q1_a.append(qfunc_b[key])

        max_qfunc_hist.append(max(q1_a))

    return max_qfunc_hist, reward_hist, track

if __name__ == "__main__":
    
    max_qfunc_hist_Q, reward_Q, track_Q = QLearning(10000, alpha = [0, 1])
    max_qfunc_hist_Double_Q, reward_Double_Q, track_Double_Q = Double_QLearning(10000, alpha = [0, 1])

    print 'QLearning:'
    print 'max Q:', max_qfunc_hist_Q
    print 'reward:', reward_Q

    print 'Double QLearning:'
    print 'max Q:', max_qfunc_hist_Double_Q
    print 'reward:', reward_Double_Q

