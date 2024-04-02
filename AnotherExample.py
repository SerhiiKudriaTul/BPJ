

import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

def parabola(x, a, b):
    return a + b * x ** 2

problem = {
    'num_vars': 2,
    'names': ['a', 'b'],
    'bounds': [[0, 1], [0, 1]]
}

param_values = saltelli.sample(problem, 2**6)

x = np.linspace(-1, 1, 100)
Y = np.array([parabola(x, *params) for params in param_values])

sobol_indices = [sobol.analyze(problem, Y_col) for Y_col in Y.T]

S1s = np.array([s['S1'] for s in sobol_indices])

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

axes[0].plot(x, S1s[:, 0], label='S1 for a')
axes[0].set_title('First-order Sobol index for parameter a')
axes[0].set_xlabel('Position x')
axes[0].set_ylabel('Sobol index')

axes[1].plot(x, S1s[:, 1], label='S1 for b')
axes[1].set_title('First-order Sobol index for parameter b')
axes[1].set_xlabel('Position x')
axes[1].set_ylabel('Sobol index')

plt.legend()
plt.show()
