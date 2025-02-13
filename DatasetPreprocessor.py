# Verbesserte Monte-Carlo-Simulation für 4 Jahre (monatlich/jährlich getrennt)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


# Funktion zum Laden und Vorbereiten der Daten
def read_and_prepare_dataset(dataset_path):
    df = pd.read_csv(dataset_path)
    df['time'] = pd.to_datetime(df['time'], format='%Y%m%d:%H%M')

    # Umbenennung der Spalten
    df = df.rename(columns={
        "G(i) (Globalstrahlung)": "global_radiation",
        "H_sun (Sonnenscheindauer in min)": "sunshine_duration",
        "T2m (Temperatur)": "temperature",
        "WS10m (Windgeschwindigkeit)": "wind_speed"
    })

    # Monatliche Gruppierung zur Analyse von Saisonalität
    df['year'] = df['time'].dt.year
    df['month'] = df['time'].dt.month

    return df


df = read_and_prepare_dataset('Timeseries_2020_2023.csv')

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ---------------------------
# Annahme: Das DataFrame 'df' existiert bereits und enthält mindestens die Spalte 'global_radiation'
# ---------------------------

# Extrahiere die global_radiation-Daten
data = df['global_radiation'].values

# ============================
# Schritt 1: Berechnung und Darstellung der PDF
# ============================
# Wir verwenden hier eine Kernel Density Estimation (KDE)
kde = stats.gaussian_kde(data)
x_values = np.linspace(np.min(data), np.max(data), 1000)
pdf_values = kde(x_values)

plt.figure(figsize=(10, 6))
plt.plot(x_values, pdf_values, label='PDF', color='blue')
plt.title('Wahrscheinlichkeitsdichtefunktion (PDF) der Global Radiation')
plt.xlabel('Global Radiation')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True)
plt.show()

# ============================
# Schritt 2: Darstellung der empirischen CDF
# ============================
# Sortiere die Daten und berechne die empirische CDF
sorted_data = np.sort(data)
cdf_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

plt.figure(figsize=(10, 6))
plt.plot(sorted_data, cdf_values, label='Empirische CDF', color='green')
plt.title('Kumulative Verteilungsfunktion (CDF) der Global Radiation')
plt.xlabel('Global Radiation')
plt.ylabel('Kumulative Wahrscheinlichkeit')
plt.legend()
plt.grid(True)
plt.show()

# ============================
# Schritt 3: Monte-Carlo-Szenario-Generierung mittels inverser CDF
# ============================
# Anzahl der zu generierenden Szenarien (Beispiel: 1000)
n_samples = 1000

# Erzeuge gleichverteilte Zufallszahlen zwischen 0 und 1
uniform_samples = np.random.uniform(0, 1, n_samples)

# Nutze np.interp zur Interpolation der quantilen (inverse CDF)
mc_scenarios = np.interp(uniform_samples, cdf_values, sorted_data)

# ============================
# Schritt 4: Visualisierung der generierten Szenarien
# ============================
plt.figure(figsize=(10, 6))
plt.hist(mc_scenarios, bins=50, density=True, alpha=0.6, label='Histogramm der MC-Szenarien', color='orange')
plt.plot(x_values, pdf_values, label='Originale PDF', color='blue', linewidth=2)
plt.title('Monte-Carlo-Szenarien für Global Radiation')
plt.xlabel('Global Radiation')
plt.ylabel('Dichte')
plt.legend()
plt.grid(True)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Angenommen, du hast bereits viele MC-Szenarien in mc_scenarios (1D-Array)
# und die ursprünglichen Szenarien sind gleich wahrscheinlich:
num_scenarios = len(mc_scenarios)
original_prob = 1.0 / num_scenarios  # da alle gleich wahrscheinlich

# 1. Bestimme, wie viele Cluster (k) du haben möchtest
k = 10  # Beispiel: 10 repräsentative Szenarien

# 2. KMeans-Clustering (Achtung: KMeans erwartet 2D-Eingaben)
mc_scenarios_2d = mc_scenarios.reshape(-1, 1)  # von (N,) zu (N,1)
kmeans = KMeans(n_clusters=k, random_state=42).fit(mc_scenarios_2d)

# 3. Cluster-Zuordnung auslesen
labels = kmeans.labels_            # welcher Punkt gehört zu welchem Cluster
cluster_centers = kmeans.cluster_centers_.flatten()  # Repräsentanten in 1D

# 4. Neu-Zuordnung von Wahrscheinlichkeiten
#    (zähle, wie viele Szenarien pro Cluster fallen)
unique, counts = np.unique(labels, return_counts=True)
counts_dict = dict(zip(unique, counts))

# Für jeden Cluster ist die Wahrscheinlichkeit die Summe der Einzelszenario-Wkt.
# Da alle Szenarien gleich wahrscheinlich waren: counts * original_prob
cluster_probs = [counts_dict[i] * original_prob for i in range(k)]

# Jetzt haben wir k repräsentative Szenarien = cluster_centers
# mit Wahrscheinlichkeiten = cluster_probs

# Optional: Sortieren nach den Werten der Globalstrahlung (der Übersicht halber)
sorted_indices = np.argsort(cluster_centers)
cluster_centers_sorted = cluster_centers[sorted_indices]
cluster_probs_sorted = np.array(cluster_probs)[sorted_indices]

# --------------------------------------------
# Visualisierung
# --------------------------------------------
plt.figure(figsize=(10, 6))

# (a) Plot aller ursprünglichen MC-Szenarien als Scatter
plt.scatter(mc_scenarios, np.zeros_like(mc_scenarios),
            alpha=0.3, label='Ursprüngliche MC-Szenarien')

# (b) Plot der Clusterzentren (reduzierte Szenarien)
plt.scatter(cluster_centers, np.zeros_like(cluster_centers),
            color='red', marker='x', s=100, label='Cluster-Repräsentanten')

plt.title('Szenarioreduktion via k-Means')
plt.xlabel('Global Radiation')
plt.yticks([])  # y-Achse ausblenden, weil wir nur 1D haben
plt.legend()
plt.grid(True)
plt.show()

# Man kann z.B. noch die Wahrscheinlichkeitsverteilung
# der reduzierten Szenarien plotten (Stems oder Balken):

plt.figure(figsize=(10, 6))
plt.stem(cluster_centers_sorted, cluster_probs_sorted, basefmt=" ")
plt.title('Reduzierte Szenarien (10 Cluster) mit Wahrscheinlichkeiten')
plt.xlabel('Global Radiation (repräsentatives Szenario)')
plt.ylabel('Wahrscheinlichkeit')
plt.grid(True)
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Angenommen, df hat die Spalten 'time' und 'global_radiation'
# Wir erstellen eine Spalte 'day_of_year'
df['day_of_year'] = df['time'].dt.dayofyear

# 1. Clustering nur auf global_radiation (1D)
X_1d = df['global_radiation'].values.reshape(-1, 1)
k = 10 # z.B. 5 Cluster
kmeans_1d = KMeans(n_clusters=k, random_state=42)
labels_1d = kmeans_1d.fit_predict(X_1d)

# 2. Scatter-Plot in 2D: Tag des Jahres vs. Globalstrahlung
plt.figure(figsize=(10,6))
scatter = plt.scatter(
    df['day_of_year'],          # x-Achse
    df['global_radiation'],     # y-Achse
    c=labels_1d,                # Cluster-Labels als Farbe
    cmap='viridis',
    alpha=0.5
)
plt.colorbar(scatter, label='Cluster Label')
plt.xlabel('Tag des Jahres')
plt.ylabel('Global Radiation')
plt.title('1D-K-Means-Cluster (Global Radiation) in 2D geplottet')
plt.grid(True)

# Wenn Du möchtest, kannst Du die 1D-Clusterzentren
# als horizontale Linien anzeigen:
centers_1d = kmeans_1d.cluster_centers_.flatten()
for center in centers_1d:
    plt.axhline(center, color='red', linestyle='--', alpha=0.7)

plt.show()

