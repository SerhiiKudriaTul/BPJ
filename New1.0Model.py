import numpy as np
import matplotlib.pyplot as plt

# Parametry simulace
N = 25  # počet buněk
T = 200  # počet časových kroků
p = 0.9  # pravděpodobnost zůstání kontaminantu v buňce, konstantní pro všechny buňky
q = 0.95  # pravděpodobnost přesunu kontaminantu do další buňky, konstantní pro všechny buňky

# Inicializace stavu systému
C = np.zeros((T, N))  # Matice kontaminantu v čase a prostoru
C[0, 0] = 1.0  # plná koncentrace v první buňce na začátku

# Simulace transportu kontaminantu včetně rozpouštění
for t in range(1, T):
    # Zůstává v buňce
    stay = C[t - 1] * p
    # Přesun do další buňky
    move = np.roll(C[t - 1] * (1 - p) * q, 1)
    move[0] = 0  # Na začátku řady se kontaminant nemůže přesunout z 'leva'
    # Rozpouštění kontaminantu do okolí
    dissolve = C[t - 1] * (1 - p) * (1 - q)

    # Aktualizace koncentrace v buňce
    C[t] = stay + move - dissolve

plt.figure(figsize=(10, 8))
plt.imshow(np.log10(C + 1e-10), aspect='auto', cmap='viridis')
plt.colorbar(label='log10(concentration)')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')
plt.title('Contaminant Transport Simulation')
plt.show()

selected_points = [0, N // 4, N // 2, 3 * N // 4, N - 1]

plt.figure(figsize=(10, 8))
for point in selected_points:
    plt.plot(range(T), C[:, point], label=f'Point {point}')

plt.xlabel('Time Step')
plt.ylabel('Concentration')
plt.title('Time Series at Selected Points')
plt.legend()
plt.show()
