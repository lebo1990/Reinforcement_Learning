import model_free_policy
import matplotlib.pyplot as plt

qfunc, J_hist = model_free_policy.MC(1000, 0.5)
print qfunc
plt.plot(J_hist)
plt.show()