import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Implementasi K-Means untuk Segmentasi Pelanggan")

# Input user
umur = st.number_input("Masukkan umur", min_value=0)
pengeluaran = st.number_input("Masukkan pengeluaran", min_value=0)

# Data training
sample = [
    [20,1],[22,2],[25,2],[30,3],
    [35,4],[40,5],[45,6],[50,7],
    
    [21,1],[23,2],[26,2],[28,3],
    [33,4],[38,5],[42,6],[48,7],
    
    [24,2],[27,3],[31,3],[36,4],
    [41,5],[44,6],[47,7],[52,7]
]

df = pd.DataFrame(sample, columns=["Umur", "Pengeluaran"])

# Model KMeans
model = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = model.fit_predict(df)

# Tampilkan data
st.subheader("Data Training")
st.write(df)

# Visualisasi
st.subheader("Visualisasi Clustering")

fig, ax = plt.subplots()
scatter = ax.scatter(df["Umur"], df["Pengeluaran"], c=df["Cluster"])

# Centroid
centroids = model.cluster_centers_
ax.scatter(centroids[:,0], centroids[:,1], marker='X', s=200)

ax.set_xlabel("Umur")
ax.set_ylabel("Pengeluaran")

st.pyplot(fig)

# Prediksi
if st.button("Prediksi Cluster"):
    data = [[umur, pengeluaran]]
    hasil = model.predict(data)
    
    st.success(f"Cluster: {hasil[0]}")

    if hasil[0] == 0:
        st.write("Kategori: Pelanggan Hemat")
    elif hasil[0] == 1:
        st.write("Kategori: Pelanggan Sedang")
    else:
        st.write("Kategori: Pelanggan Boros")
