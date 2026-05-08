import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans

st.title("Clustering Pelanggan (K-Means)")

# Input user
umur = st.number_input("Masukkan umur", min_value=0)
pengeluaran = st.number_input("Masukkan pengeluaran", min_value=0)

if st.button("Prediksi Cluster"):
    data = [[umur, pengeluaran]]
    
    # Data training
    sample = [
        [20,1],[22,2],[25,2],[30,3],
        [35,4],[40,5],[45,6],[50,7]
    ]
    
    # Model KMeans
    model = KMeans(n_clusters=3, random_state=42)
    model.fit(sample)
    
    hasil = model.predict(data)
    
    st.success(f"Cluster: {hasil[0]}")