import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol
import time

# Parametry simulace
N = 40  # počet buněk
T = 200  # počet časových kroků

# Rozsahy pro 'p' a 'q' jako části zadání
bounds_p = [[0.5, 0.6]]
bounds_q = [[0.9, 1.0]]

problem = {
    'num_vars': 2,
    'names': ['P', 'Q'],
    'bounds': bounds_p + bounds_q
}

param_values = saltelli.sample(problem, 1024)

# Funkce pro spuštění simulace
def run_simulation(params):
    p = params[0]
    q = params[1]
    C = np.zeros((T, N))
    C[0, 0] = 1.0

    for t in range(1, T):
        C_prev = C[t - 1].copy()
        for i in range(N):
            if i == 0:
                C[t, i] = C_prev[i] * p
            else:
                C[t, i] = C_prev[i] * p + C_prev[i - 1] * (1 - p) * q

    return C[-1, -1]  # Koncentrace na pravém konci oblasti

# Začátek měření času
start_time = time.time()

# Výpočet výstupů modelu pro každý vzorek
Y = np.array([run_simulation(v) for v in param_values])

# Konec měření času
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Výpočet vzorků trval {elapsed_time:.2f} sekund.")

# Analýza citlivosti
Si = sobol.analyze(problem, Y, print_to_console=True)

# Vykreslení výsledků analýzy citlivosti
fig, ax = plt.subplots(figsize=(8, 6))

bar_width = 0.4
indices = np.arange(len(problem['names']))

# S1 and ST bar plot
bars_S1 = ax.barh(indices, Si['S1'], bar_width, xerr=Si['S1_conf'], capsize=5, color='blue', alpha=0.7, label='S1')
bars_ST = ax.barh(indices, Si['ST'], bar_width, xerr=Si['ST_conf'], capsize=5, color='red', alpha=0.3, hatch='//', label='ST')

ax.set_yticks(indices)
ax.set_yticklabels(problem['names'])
ax.set_title('Sobol Indices with Confidence Intervals')
ax.set_xlabel('Sobol Index')
ax.legend()

# Přidání hodnot vedle pruhů
def autolabel(bars, values, offset=0):
    for bar, value in zip(bars, values):
        width = bar.get_width()
        ax.annotate(f'{value:.2f}',
                    xy=(width, bar.get_y() + bar.get_height() / 2 + offset),
                    xytext=(3, 0),  # 3 points horizontal offset
                    textcoords="offset points",
                    ha='left', va='center')

autolabel(bars_S1, Si['S1'], offset=-0.15)
autolabel(bars_ST, Si['ST'], offset=0.15)

plt.show()
