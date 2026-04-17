import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import zipfile
import requests
import gdown


# ==========================================
# 1. PAGE CONFIG & CUSTOM STYLING
# ==========================================
st.set_page_config(page_title="Fashion AI | Myntra", layout="wide")

st.markdown("""
    <style>
    /* Main Background */
    .stApp { background-color: #F5E8D8; }
    
    /* Modern Card UI */
    .fashion-card {
        background-color: white;
        padding: 15px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        text-align: center;
    }
    
    /* Query Badge */
    .query-header {
        background: linear-gradient(90deg, #FF4B4B, #FF8E53);
        color: white;
        padding: 8px 15px;
        border-radius: 8px;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 10px;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #8A9A5B;
        
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. DATA & MODEL CLASSES
# ==========================================
class FeatureExtractor:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Using ResNet-18 as requested
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model = nn.Sequential(*list(base_model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract(self, image):
        image = image.convert('RGB')
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(tensor)
        return features.squeeze().cpu().numpy()

class FashionRecommender:
    def __init__(self, features, metadata):
        self.features = features
        self.metadata = metadata
        self.knn = NearestNeighbors(n_neighbors=20, metric='cosine', algorithm='brute')
        self.knn.fit(features)

    def get_similar(self, query_features, n=6):
        query_features = query_features.reshape(1, -1)
        distances, indices = self.knn.kneighbors(query_features, n_neighbors=n)
        
        results = self.metadata.iloc[indices[0]].copy()
        results['similarity'] = 1 - distances[0]
        return results

# ==========================================
# 3. UTILITY FUNCTIONS (Caching & Loading)
# ==========================================
@st.cache_resource
def load_assets():
    # Load your trained PKL file
    if not os.path.exists('models/fashion_recommender.pkl'):
        st.error("Error: 'models/fashion_recommender.pkl' not found. Please train the model first.")
        st.stop()
    
    with open('models/fashion_recommender.pkl', 'rb') as f:
        data = joblib.load(f)
    
    # Path fix for different OS
    data['metadata']['image_path'] = data['metadata']['image_path'].str.replace('\\', '/', regex=False)
    
    extractor = FeatureExtractor()
    recommender = FashionRecommender(data['features'], data['metadata'])
    return extractor, recommender, data['metadata']

# ==========================================
# 4. UI MODES
# ==========================================
def show_catalog(extractor, recommender, metadata):
    st.markdown("### 🏷️ Explore Myntra Catalog")
    
    if st.button("Shuffle Catalog"):
        st.session_state.samples = metadata.sample(10)
    
    if 'samples' not in st.session_state:
        st.session_state.samples = metadata.sample(10)

    cols = st.columns(5)
    selected_idx = None
    for i, (_, row) in enumerate(st.session_state.samples.iterrows()):
        with cols[i % 5]:
            st.image(row['image_path'], use_container_width=True)
            if st.button(f"Find Similar", key=f"btn_{row.name}"):
                selected_idx = row.name

    if selected_idx is not None:
        st.write("---")
        display_results(metadata.iloc[selected_idx]['image_path'], 
                        recommender.get_similar(recommender.features[metadata.index.get_loc(selected_idx)]))

def show_upload(extractor, recommender):
    st.markdown("### 📤 Upload Your Style")
    file = st.file_uploader("Upload an image (JPG/PNG)", type=['jpg','jpeg','png'])
    
    if file:
        img = Image.open(file)
        with st.spinner("AI is analyzing textures and patterns..."):
            features = extractor.extract(img)
            recommendations = recommender.get_similar(features)
            display_results(img, recommendations, is_path=False)

def display_results(query_img, recs, is_path=True):
    col1, col2 = st.columns([1, 3], gap="large")
    
    with col1:
        st.markdown('<div class="query-header">QUERY ITEM</div>', unsafe_allow_html=True)
        st.image(query_img, use_container_width=True)
        
    with col2:
        st.markdown("### 🌟 AI Recommendations")
        # Filter out the query item if it's in the results (similarity near 100%)
        recs = recs[recs['similarity'] < 0.99].head(6)
        
        cols = st.columns(3)
        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 3]:
                st.markdown('<div class="fashion-card">', unsafe_allow_html=True)
                st.image(row['image_path'], use_container_width=True)
                st.markdown(f"**Match: {row['similarity']:.1%}**")
                st.progress(float(row['similarity']))
                st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================
def main():
    st.markdown('<h1 style="text-align:center; color:#1E1E1E;">MYNTRA <span style="color:#FF4B4B;">AI-SEARCH</span></h1>', unsafe_allow_html=True)
    
    extractor, recommender, metadata = load_assets()
    
    st.sidebar.title("🎮 Navigation")
    mode = st.sidebar.radio("Choose Mode", ["Browse Catalog", "Upload Image", "System Analytics"])
    
    if mode == "Browse Catalog":
        show_catalog(extractor, recommender, metadata)
    elif mode == "Upload Image":
        show_upload(extractor, recommender)
    else:
        st.subheader("📊 Model Stats")
        st.metric("Total Items Indexed", f"{len(metadata):,}")
        st.metric("Feature Dimensions", "512 (ResNet-18)")
        st.dataframe(metadata.head(20))

if __name__ == "__main__":
    main()
