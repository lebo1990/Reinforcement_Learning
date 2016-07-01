import unittest

from Model_based_value_iteration import *


class policy_iteration_test(unittest.TestCase):

    def test_policy_evaluation(self):
        grid_mdp = MDP()
        policy_value = Policy_Value(grid_mdp)
        policy_value.value_iteration(grid_mdp)

if __name__ == '__main__':
    unittest.main()
