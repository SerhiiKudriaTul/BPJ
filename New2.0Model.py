import numpy as np
import matplotlib.pyplot as plt

# Parametry simulace
N = 25  # počet buněk
T = 200  # počet časových kroků
p = 0.5  # pravděpodobnost zůstání kontaminantu v buňce, konstantní pro všechny buňky
q = 0.95  # pravděpodobnost přesunu kontaminantu do další buňky, konstantní pro všechny buňky

#parametry udelat jako vektory N, T, p, q zvysit pocet vektoru, 3 useky 300 prvky prku a vic
# C[N] posledni hodnota bude zaviset na P1, P2, P3 a pak treba s Q1, Q2, Q3

# Inicializace stavu systému
C = np.zeros((T, N))  # Matice kontaminantu v čase a prostoru
C[0, 0] = 1.0  # plná koncentrace v první buňce na začátku

# Vektorizovaná simulace transportu kontaminantu
for t in range(1, T):
    C_prev = C[t - 1].copy()
    C[t, 1:-1] = C_prev[1:-1]*p + C_prev[0:-2]*(1-p)*q
    # Přesun do další buňky
    C[t, -1] = C_prev[-1]*p  # Na konci řady se kontaminant nemůže přesunout dál

# Vizualizace pomocí časoprostorové matice
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(C + 1e-10), aspect='auto', cmap='viridis')
plt.colorbar(label='log10(concentration)')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')
plt.title('Contaminant Transport Simulation')
plt.show()

selected_points = [0, N//4, N//2, 3*N//4, N-1]

plt.figure(figsize=(10, 8))
for point in selected_points:
    plt.plot(range(T), C[:, point], label=f'Point {point}')

plt.xlabel('Time Step')
plt.ylabel('Concentration')
plt.title('Time Series at Selected Points')
plt.legend()
plt.show()
