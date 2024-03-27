import numpy as np
import matplotlib.pyplot as plt

# Parametry simulace
N = 25  # počet buněk
T = 200  # počet časových kroků
p = 0.9  # pravděpodobnost zůstání kontaminantu v buňce
q = 0.95  # pravděpodobnost přesunu kontaminantu do další buňky

# Inicializace stavu systému
C = np.zeros((T, N))  # Matice kontaminantu v čase a prostoru
C[0, 0] = 1.0  # plná koncentrace v první buňce na začátku

# Simulace transportu kontaminantu
for t in range(1, T):
    for i in range(N):
        # Zůstává v buňce
        if i == 0:
            stay = C[t - 1, i] * p
        else:
            stay = C[t - 1, i] * p + C[t - 1, i - 1] * (1 - p) * q

        # Přesun do další buňky nebo rozpuštění do okolní skály
        if i < N - 1:
            move = C[t - 1, i] * (1 - p) * q
        else:
            move = 0  # Na konci řady se kontaminant nemůže přesunout dál

        dissolve = C[t - 1, i] * (1 - p) * (1 - q)

        # Aktualizace koncentrace v buňce
        C[t, i] = stay + move

# Vizualizace pomocí časoprostorové matice
plt.figure(figsize=(10, 8))
plt.imshow(np.log10(C + 1e-10), aspect='auto', cmap='viridis')
plt.colorbar(label='log10(concentration)')
plt.xlabel('Cell Index')
plt.ylabel('Time Step')
plt.title('Contaminant Transport Simulation')
plt.show()

# Seznam indexů vybraných bodů
selected_points = [0, N//2, N-1, 3*N//8, 5*N//8]

# Vytvoření grafů časové závislosti pro vybrané body
plt.figure(figsize=(10, 8))
for point in selected_points:
    plt.plot(range(T), C[:, point], label=f'Point {point}')

plt.xlabel('Časový krok')
plt.ylabel('Koncentrace')
plt.title('Grafy časové závislosti pro vybrané body')
plt.legend()
plt.show()


