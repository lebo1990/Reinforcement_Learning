import random
#random.seed(0)
import matplotlib.pyplot as plt
from model_free import *
from policy import *
from evaluate import *
from MDP_2 import *

if __name__ == "__main__":
    
    plt.close('all')
    grid_wall = MDP_wall()
    policy_wall = Policy(grid_wall, epsilon=0.5)
    evaler_wall = Evaler(grid_wall, file_name = "./bestQ_MDP_2.txt")
    grid_ID = MDP_ID()
    policy_ID = Policy(grid_ID, epsilon=0.5)
    evaler_ID = Evaler(grid_ID, file_name = "./bestQ_MDP_2.txt")

    iterations = 1000
    alpha_1 = 0.2

    plt.figure(1)
    # MC
    policy_wall_MC, J_MC1 = MC(
        grid_wall, policy_wall, evaler_wall, num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_MC1, label="MC wall feature")

    policy_ID_MC, J_MC2 = MC(
        grid_ID, policy_ID, evaler_ID, num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_MC2, label="MC ID feature")

    # SARSA
    policy_ID_SARSA, J_SARSA2 = SARSA(grid_ID, policy_ID, evaler_ID,
        num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_SARSA2, "--", label="SARSA ID feature")

    # Qlearning

    policy_ID_Qlearning, J_Qlearning2 = Qlearning(grid_ID, policy_ID, evaler_ID,
        num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_Qlearning2, "-.", label="Q-learning ID feature")
    plt.title("Convergence")
    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.legend(loc='best')

    plt.figure(2)
    policy_wall_SARSA, J_SARSA1 = SARSA(
        grid_wall, policy_wall, evaler_wall, num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_SARSA1, "--", label="SARSA wall feature")
    policy_wall_Qlearning, J_Qlearning1 = Qlearning(
        grid_wall, policy_wall, evaler_wall, num_iter1=iterations, alpha=alpha_1)
    plt.plot(J_Qlearning1, "-.", label="Q-learning wall feature")
    plt.xlabel("number of iterations")
    plt.ylabel("square errors")
    plt.title("No convergence")
    plt.legend(loc='best')

    plt.show()
