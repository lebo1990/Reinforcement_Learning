# import numpy
import sys
import random
# random.seed(0)


class MDP:

    def __init__(self):

        self.states = [1, 2, 3, 4, 5, 6, 7, 8]  # 0 indicates end
        self.terminal_states = dict()
        self.terminal_states[6] = 1
        self.terminal_states[7] = 1
        self.terminal_states[8] = 1

        self.actions = ['n', 'e', 's', 'w']

        self.rewards = dict()
        self.rewards['1_s'] = -1.0
        self.rewards['3_s'] = 1.0
        self.rewards['5_s'] = -1.0

        self.t = dict()
        self.t['1_s'] = 6
        self.t['1_e'] = 2
        self.t['2_w'] = 1
        self.t['2_e'] = 3
        self.t['3_s'] = 7
        self.t['3_w'] = 2
        self.t['3_e'] = 4
        self.t['4_w'] = 3
        self.t['4_e'] = 5
        self.t['5_s'] = 8
        self.t['5_w'] = 4

        self.gamma = 0.8

    def transform(self, state, action):  # return is_terminal,state, reward
        if state in self.terminal_states:
            return True, state, 0

        key = '%d_%s' % (state, action)
        if key in self.t:
            next_state = self.t[key]
        else:
            next_state = state

        is_terminal = False
        if next_state in self.terminal_states:
            is_terminal = True

        if key not in self.rewards:
            r = 0.0
        else:
            r = self.rewards[key]
        return is_terminal, next_state, r


class Policy_Value:
    def __init__(self, grid_mdp, iterations=1000, error=1e-6):
        self.iterations = iterations
        self.error = error
        self.v = [0.0 for i in xrange(len(grid_mdp.states) + 1)]
        self.pi = dict()
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states:
                continue
            self.pi[state] = grid_mdp.actions[0]

    def policy_iterate(self, grid_mdp):
        for i in xrange(self.iterations):
            self.policy_evaluate(grid_mdp)
            self.policy_improve(grid_mdp)

    def policy_evaluate(self, grid_mdp):
        while True:
            delta = 0.0
            for state in grid_mdp.states:
                if state in grid_mdp.terminal_states:
                    continue
                action = self.pi[state]
                t, s, r = grid_mdp.transform(state, action)
                new_v = r + grid_mdp.gamma * self.v[s]
                # delta = max(delta, abs(self.v[state] - new_v))
                delta += abs(self.v[state] - new_v)
                self.v[state] = new_v

            if delta < self.error:
                break

    def policy_improve(self, grid_mdp):
        for state in grid_mdp.states:
            if state in grid_mdp.terminal_states:
                continue
            a1 = grid_mdp.actions[0]
            t, s, r = grid_mdp.transform(state, a1)
            v1 = r + grid_mdp.gamma * self.v[s]

            for action in grid_mdp.actions:
                t, s, r = grid_mdp.transform(state, action)
                if v1 < r + grid_mdp.gamma * self.v[s]:
                    a1 = action
                    v1 = r + grid_mdp.gamma * self.v[s]
            self.pi[state] = a1
#            print state, self.pi[state]

if __name__ == "__main__":
    for iter in xrange(1, 10):
        grid_mdp = MDP()
        policy_value = Policy_Value(grid_mdp, iterations=iter)
        policy_value.policy_iterate(grid_mdp)
        print 'Iteration %d:' % iter
        print "policy:"
        for i in xrange(1, 6):
            print "%d->%s\t" % (i, policy_value.pi[i]),
        print

        print "value:"
        for i in xrange(1, 6):
            print "%d:%f\t" % (i, policy_value.v[i]),
        print
        print
