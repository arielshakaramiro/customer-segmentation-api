import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
import os


# Membuat dataset dengan 3 cluster
X, _ = make_blobs(n_samples=200, centers=3, cluster_std=1.05, random_state=42)

# Konversi ke DataFrame untuk kemudahan manipulasi
df = pd.DataFrame(X, columns=["Annual Spending (USD)", "Purchase Frequency"])

# Normalisasi data menggunakan StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# Mencari jumlah cluster optimal dengan Metode Elbow
inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Menyusun hasil dalam bentuk DataFrame
elbow_results = pd.DataFrame({"Jumlah Cluster (K)": list(K_range), "Inertia": inertia})
print(elbow_results)

# Plot Metode Elbow
plt.figure(figsize=(8, 5))
plt.plot(K_range, inertia, marker="o", linestyle="-")
plt.xlabel("Jumlah Cluster (K)")
plt.ylabel("Inertia")
plt.title("Metode Elbow untuk Menentukan K")
plt.grid()
plt.show()

plt.savefig("outputs/elbow_plot.png")


# Menerapkan K-Means dengan 3 cluster
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Mendapatkan pusat cluster
centroids = kmeans.cluster_centers_

# Visualisasi hasil clustering
plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=df["Cluster"], cmap="viridis", alpha=0.6)
plt.scatter(centroids[:, 0], centroids[:, 1], c="red", marker="X", s=200, label="Centroids")
plt.xlabel("Annual Spending (Normalized)")
plt.ylabel("Purchase Frequency (Normalized)")
plt.title("Segmentasi Pelanggan dengan K-Means")
plt.legend()
plt.grid()
plt.show()

plt.savefig("outputs/cluster_plot.png")


# Buat folder models jika belum ada
os.makedirs("models", exist_ok=True)

# Simpan model dan scaler ke file
with open("models/kmeans_model.pkl", "wb") as file:
    pickle.dump(kmeans, file)

with open("models/kmeans_scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)

print("Model dan scaler berhasil disimpan di folder 'models'.")