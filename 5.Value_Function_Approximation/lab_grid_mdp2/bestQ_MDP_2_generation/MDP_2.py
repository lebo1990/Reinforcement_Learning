# Small grid world MDP 2
#
#  __________________________
# |        |        |        |
# |  (7)   |  (8)   |  (9)   |
# |        |        |  Stop  |
# |________|________|________|
# |        |        |        |
# |  (4)   |  (5)   |  (6)   |
# |        |        |        |
# |________|________|________|
# |        |        |        |
# |  (1)   |  (2)   |  (3)   |
# | Start  |        |        |
# |________|________|________|
# 
# Reward(S: 9) = +5
# Other reward = 0



import random


class MDP:

    def __init__(self):

        self.states = [1, 2, 3, 4, 5, 6, 7, 8, 9]  # 0 indicates end
        self.terminal_states = dict()
        self.terminal_states[9] = 1

        self.actions = ['n', 'e', 's', 'w']

        self.rewards = dict()
        self.rewards['6_n'] = 5.0
        self.rewards['8_e'] = 5.0
        

        self.t = dict()
        self.t['1_n'] = 4
        self.t['1_e'] = 2

        self.t['2_n'] = 5
        self.t['2_e'] = 3
        self.t['2_w'] = 1

        self.t['3_n'] = 6
        self.t['3_w'] = 2

        self.t['4_n'] = 7
        self.t['4_e'] = 5
        self.t['4_s'] = 1

        self.t['5_n'] = 8
        self.t['5_e'] = 6
        self.t['5_s'] = 2
        self.t['5_w'] = 4

        self.t['6_n'] = 9
        self.t['6_s'] = 3
        self.t['6_w'] = 5

        self.t['7_e'] = 8
        self.t['7_s'] = 4

        self.t['8_e'] = 9
        self.t['8_s'] = 5
        self.t['8_w'] = 7

        self.gamma = 0.8

    def getTerminal(self):
        return self.terminal_states

    def getGamma(self):
        return self.gamma

    def getStates(self):
        return self.states

    def getActions(self):
        return self.actions

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
            r = 0
        else:
            r = self.rewards[key]

        return is_terminal, next_state, r

    def gen_randompi_sample(self, num):
        state_sample = []
        action_sample = []
        reward_sample = []
        for i in xrange(num):
            s_tmp = []
            a_tmp = []
            r_tmp = []

            s = random.choice(self.states)
            t = False
            while t == False:
                a = random.choice(self.actions)
                t, s1, r = self.transform(s, a)
                s_tmp.append(s)
                r_tmp.append(r)
                a_tmp.append(a)
                s = s1
            state_sample.append(s_tmp)
            reward_sample.append(r_tmp)
            action_sample.append(a_tmp)

        return state_sample, action_sample, reward_sample
