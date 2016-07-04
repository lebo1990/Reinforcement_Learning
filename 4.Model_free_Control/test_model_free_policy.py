from model_free_policy import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    plt.figure(1)
    iterations = 5000
    alpha = 0.2
    epsilon = 0.2
    for itera in xrange(1, 5):
        qfunc_MC, J_hist_MC = MC(iterations, epsilon)
        plt.plot(J_hist_MC, '--', label='MC, iteration %d' % itera)
        qfunc_SARSA, J_hist_SARSA = SARSA(iterations, alpha, epsilon)
        plt.plot(J_hist_SARSA, ':', label='SARSA, iteration %d' % itera)
        qfunc_QLearning, J_hist_QLearning = QLearning(iterations, alpha, epsilon)
        plt.plot(J_hist_QLearning, label='QLearning, iteration %d' % itera)
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Conclusion: SARSA is the most random one, Q-learning is stable')

    plt.figure(2)
    iterations = 5000
    EPSILON = [0.2, 0.5, 0.8]
    ALPHA = [0.2]
    for epsilon in EPSILON:
        qfunc_MC, J_hist_MC = MC(iterations, epsilon)
        plt.plot(J_hist_MC, '--', label='MC, epsilon = %.2f' % epsilon)
        for alpha in ALPHA:
            qfunc_SARSA, J_hist_SARSA = SARSA(iterations, alpha, epsilon)
            plt.plot(J_hist_SARSA, ':', label='SARSA, alpha = %.2f, epsilon = %.2f' %
                     (alpha, epsilon))
            qfunc_QLearning, J_hist_QLearning = QLearning(
                iterations, alpha, epsilon)
            plt.plot(J_hist_QLearning, label='QLearning, alpha = %.2f, epsilon = %.2f' %
                     (alpha, epsilon))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Conclusion: Q-learning is off-policy, SARSA is unstable')

    plt.figure(3)
    iterations = 5000
    EPSILON = [0.2, 0.5]
    ALPHA = [0.2, 0.4]
    for epsilon in EPSILON:
        qfunc_MC, J_hist_MC = MC(iterations, epsilon)
        plt.plot(J_hist_MC, '--', label='MC, epsilon = %.2f' % epsilon)
        for alpha in ALPHA:
            qfunc_SARSA, J_hist_SARSA = SARSA(iterations, alpha, epsilon)
            plt.plot(J_hist_SARSA, ':', label='SARSA, alpha = %.2f, epsilon = %.2f' %
                     (alpha, epsilon))
            qfunc_QLearning, J_hist_QLearning = QLearning(
                iterations, alpha, epsilon)
            plt.plot(J_hist_QLearning, label='QLearning, alpha = %.2f, epsilon = %.2f' %
                     (alpha, epsilon))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Conclusion: Q-learning is the best one')

    plt.figure(4)
    iterations = 5000
    ALPHA = [0.1, 0.3, 0.5, 0.8]
    epsilon = 0.2
    qfunc_MC, J_hist_MC = MC(iterations, epsilon)
    plt.plot(J_hist_MC, '--', label='MC, epsilon = %.2f' % epsilon)
    for alpha in ALPHA:
        qfunc_SARSA, J_hist_SARSA = SARSA(iterations, alpha, epsilon)
        plt.plot(J_hist_SARSA, ':', label='SARSA, alpha = %.2f, epsilon = %.2f' %
                 (alpha, epsilon))
        qfunc_QLearning, J_hist_QLearning = QLearning(
            iterations, alpha, epsilon)
        plt.plot(J_hist_QLearning, label='QLearning, alpha = %.2f, epsilon = %.2f' %
                 (alpha, epsilon))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Conclusion: Larger learning rate alpha make it faster convergence')

    plt.figure(5)
    iterations = 5000
    ALPHA = [0.1, 0.3, 0.5, 0.8]
    epsilon = 0.2
    qfunc_MC, J_hist_MC = MC(iterations, epsilon)
    plt.plot(J_hist_MC, '--', label='MC, epsilon = %.2f' % epsilon)
    for alpha in ALPHA:
        qfunc_SARSA, J_hist_SARSA = SARSA(iterations, alpha, epsilon)
        plt.plot(J_hist_SARSA, ':', label='SARSA, alpha = %.2f, epsilon = %.2f' %
                 (alpha, epsilon))
        qfunc_QLearning, J_hist_QLearning = QLearning(
            iterations, alpha, epsilon)
        plt.plot(J_hist_QLearning, label='QLearning, alpha = %.2f, epsilon = %.2f' %
                 (alpha, epsilon))
    plt.legend()
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.title('Conclusion: learning rate')

    plt.show()
