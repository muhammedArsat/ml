import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data
data = pd.read_csv('patient_data.csv')

features = data[['Age', 'Blood Pressure', 'Cholesterol', 'Blood Sugar', 'Height', 'Weight', 'BMI', 'Pulse Rate', 
                 'Respiratory Rate', 'Oxygen Saturation', 'Creatinine Level', 'Hemoglobin Level']]

# Scale features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Clustering functions
def kmeans_clustering(n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(scaled_features)
    return kmeans.labels_

def em_clustering(n_components):
    em = GaussianMixture(n_components=n_components, random_state=42)
    em.fit(scaled_features)
    return em.predict(scaled_features)

def evaluate_clustering(labels):
    silhouette = silhouette_score(scaled_features, labels)
    calinski = calinski_harabasz_score(scaled_features, labels)
    return silhouette, calinski

def visualize_clusters(labels):
    pca = PCA(n_components=2)
    pca_features = pca.fit_transform(scaled_features)
    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title('Cluster Visualization')
    return plt

# App Title
st.title("🩺 Patient Clustering for Personalized Treatment Plans")

# Sidebar for cluster settings
st.sidebar.subheader("Cluster Settings")
n_clusters = st.sidebar.slider("Select Number of Clusters (K-Means)", 2, 10, 5)
n_components = st.sidebar.slider("Select Number of Components (EM)", 2, 10, 5)

# Perform clustering
kmeans_labels = kmeans_clustering(n_clusters)
em_labels = em_clustering(n_components)

# Evaluate clustering
kmeans_silhouette, kmeans_calinski = evaluate_clustering(kmeans_labels)
em_silhouette, em_calinski = evaluate_clustering(em_labels)

# Tabs for results comparison
st.subheader("Clustering Results Comparison")

tab1, tab2 = st.tabs(["K-Means Clustering", "EM Clustering"])

# Tab 1: K-Means
with tab1:
    st.write("#### K-Means Clustering")
    st.write(f"**Silhouette Score:** {kmeans_silhouette:.3f}")
    st.write(f"**Calinski-Harabasz Index:** {kmeans_calinski:.3f}")
    
    # Visualization for K-Means
    st.write("#### Cluster Visualization (K-Means)")
    plt.figure(figsize=(5, 5))
    plt = visualize_clusters(kmeans_labels)
    st.pyplot(plt)

# Tab 2: EM Clustering
with tab2:
    st.write("#### Expectation-Maximization (EM) Clustering")
    st.write(f"**Silhouette Score:** {em_silhouette:.3f}")
    st.write(f"**Calinski-Harabasz Index:** {em_calinski:.3f}")
    
    # Visualization for EM
    st.write("#### Cluster Visualization (EM)")
    plt.figure(figsize=(5, 5))
    plt = visualize_clusters(em_labels)
    st.pyplot(plt)

# Conclusion section with enhanced text
st.subheader("Conclusion")

if kmeans_silhouette > em_silhouette and kmeans_calinski > em_calinski:
    st.write("🔹 **K-Means performs better** based on both the Silhouette Score and the Calinski-Harabasz Index.")
elif em_silhouette > kmeans_silhouette and em_calinski > kmeans_calinski:
    st.write("🔹 **Expectation-Maximization (EM) performs better** based on both the Silhouette Score and the Calinski-Harabasz Index.")
else:
    st.write("🔸 The performance is mixed:")
    st.write(f"- **K-Means Silhouette Score:** {kmeans_silhouette:.3f}, **Calinski-Harabasz Index:** {kmeans_calinski:.3f}")
    st.write(f"- **EM Silhouette Score:** {em_silhouette:.3f}, **Calinski-Harabasz Index:** {em_calinski:.3f}")

# Customize the theme (optional in Streamlit settings)
