import numpy as np
import matplotlib.pyplot as plt
from SALib.sample import saltelli
from SALib.analyze import sobol

# Parametry simulace
N = 40  # počet buněk
T = 100  # počet časových kroků

# Rozsahy pro 'p' a 'q' jako části zadání
bounds_p = [[0.3, 0.5], [0.55, 0.65], [0.75, 0.85]]
bounds_q = [[0.9, 0.95], [0.85, 0.9], [0.8, 0.85]]

problem = {
    'num_vars': 6,
    'names': ['P1', 'P2', 'P3', 'Q1', 'Q2', 'Q3'],
    'bounds': bounds_p + bounds_q
}

param_values = saltelli.sample(problem, 4)

# Funkce pro spuštění simulace
def run_simulation(params):
    p_sampled = np.concatenate((np.full(N // 3, params[0]), np.full(N // 3, params[1]), np.full(N // 3, params[2])))
    q_sampled = np.concatenate((np.full(N // 3, params[3]), np.full(N // 3, params[4]), np.full(N // 3, params[5])))
    C = np.zeros((T, N))
    C[0, 0] = 1.0

    for t in range(1, T):
        C_prev = C[t - 1].copy()
        for i in range(1, N-1):
            C[t, i] = C_prev[i]*p_sampled[i] + C_prev[i-1]*(1-p_sampled[i])*q_sampled[i]
        C[t, -1] = C_prev[-1]*p_sampled[-1]  # Na konci řady se kontaminant nemůže přesunout dál

    # Vizualizace pomocí časoprostorové matice
    plt.figure(figsize=(10, 8))
    plt.imshow(np.log10(C + 1e-10), aspect='auto', cmap='viridis')
    plt.colorbar(label='log10(concentration)')
    plt.xlabel('Cell Index')
    plt.ylabel('Time Step')
    plt.title('Contaminant Transport Simulation')

    # Vizualizace časové řady pro vybrané body
    selected_points = [0, N//4, N//2, 3*N//4, N-1]
    plt.figure(figsize=(10, 8))
    for point in selected_points:
        plt.plot(range(T), C[:, point], label=f'Point {point}')
    plt.xlabel('Time Step')
    plt.ylabel('Concentration')
    plt.title('Time Series at Selected Points')
    plt.legend()

    return np.max(C)

# Výpočet výstupů modelu pro každý vzorek
Y = np.array([run_simulation(v) for v in param_values])

# Analýza citlivosti
Si = sobol.analyze(problem, Y, print_to_console=False)

# Vykreslení výsledků analýzy citlivosti
if 'S1' in Si:
    indices = Si['S1']
    errors = Si['S1_conf']
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(indices)), indices, yerr=np.abs(errors.T[0] - indices), capsize=5, color='blue', alpha=0.7)
    plt.xticks(range(len(indices)), problem['names'])
    plt.title('First Order Sobol Indices with Confidence Intervals')
    plt.ylabel('Sobol Index')
    plt.show()
else:
    print("Sobol indices were not calculated.")

plt.close('all')
