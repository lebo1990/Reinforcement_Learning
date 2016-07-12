import random
# random.seed(0)
import matplotlib.pyplot as plt
from gradient import *
from policy_value import *
from MDP_2 import *
from evaluate import *


if __name__ == "__main__":

    iterations = 10000
    alpha1 = 0.2
    epsilon1 = 0.2

    plt.close('all')

    grid_wall = grid_mdp_wall()
    softmaxpolicy_wall = SoftmaxPolicy(grid_wall, epsilon=epsilon1)
    valuepolicy_wall = ValuePolicy(grid_wall, epsilon=epsilon1)
    evaler_wall = Evaler(grid_wall)


    grid_ID = grid_mdp_ID()
    softmaxpolicy_ID = SoftmaxPolicy(grid_ID, epsilon=epsilon1)
    valuepolicy_ID = ValuePolicy(grid_ID, epsilon=epsilon1)
    evaler_ID = Evaler(grid_ID)


    # SARSA
    softmaxpolicy_wall_SARSA, J_wall_SARSA = SARSA(
        grid_wall, evaler_wall, softmaxpolicy_wall, valuepolicy_wall, iterations, alpha = alpha1)
    plt.plot(J_wall_SARSA, "-", label="SARSA wall feature")
    softmaxpolicy_ID_SARSA, J_ID_SARSA = SARSA(
        grid_ID, evaler_ID, softmaxpolicy_ID, valuepolicy_ID, iterations, alpha = alpha1)
    plt.plot(J_ID_SARSA, "-", label="SARSA ID feature")

    # Q-learning
    softmaxpolicy_wall_Q, J_wall_Q = Qlearning(
        grid_wall, evaler_wall, softmaxpolicy_wall, valuepolicy_wall, iterations, alpha = alpha1)
    plt.plot(J_wall_Q, "-.", label="Q-learning wall feature")
    softmaxpolicy_ID_Q, J_ID_Q = Qlearning(
        grid_ID, evaler_ID, softmaxpolicy_ID, valuepolicy_ID, iterations, alpha = alpha1)
    plt.plot(J_ID_Q, "-.", label="Q-learning ID feature")

    plt.xlabel("No. of iterations")
    plt.ylabel("Loss")
    plt.legend(loc='best')
    # plt.title('Conclusion')
    plt.show()
