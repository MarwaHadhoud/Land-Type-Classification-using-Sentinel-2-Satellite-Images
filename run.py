import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from skimage.util import img_as_ubyte
from sklearn.metrics import classification_report , confusion_matrix
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, exposure, transform
from PIL import Image
import os
import json

# Load models
@st.cache_resource
def load_class_labels():
    with open('label_map.json') as f:
        return json.load(f)

def load_models():
    return {
        'svm': load('models/svm_model.joblib'),
        'scaler': load('models/svm_scaler.joblib'),
        'rf': load('models/rf_model.joblib'),
        'feature_names': load('models/feature_names.joblib'),
        'class_labels': load_class_labels()  
  
    }
# functions that used for extract features 
def summarize_array_column(df, column_name):
    df[f"{column_name}_mean"] = df[column_name].apply(np.mean)
    df[f"{column_name}_std"] = df[column_name].apply(np.std)
    df[f"{column_name}_min"] = df[column_name].apply(np.min)
    df[f"{column_name}_max"] = df[column_name].apply(np.max)
    return df.drop(column_name, axis=1)

# function to extract features
@st.cache_data
def extract_advanced_features(image_path, bins=32, distances=[1], angles=[0]):
    """
    Extract comprehensive features from RGB images including:
    - Statistical features (mean, std, etc.)
    - HSV components
    - Color histograms
    - Texture features (GLCM)
    - RGB-based vegetation indices
    
    Args:
        image_path: Path to the RGB image (uint8)
        bins: Number of bins for histograms
        distances/angles: GLCM parameters
        
    Returns:
        Dictionary of features
    """
    # Load and validate image
    img = io.imread(image_path)
    if img.dtype != np.uint8:
        img = img_as_ubyte(img)
    R, G, B = img[..., 0], img[..., 1], img[..., 2]
    features = {}
    
    # ======================
    # 1. BASIC STATISTICS
    # ======================
    for channel, name in zip([R, G, B], ['red', 'green', 'blue']):
        features[f'{name}_mean'] = np.mean(channel)
        features[f'{name}_std'] = np.std(channel)
        features[f'{name}_median'] = np.median(channel)
        features[f'{name}_skew'] = pd.Series(channel.flatten()).skew()
    
    # ======================
    # 2. HSV STATISTICS
    # ======================
    hsv = color.rgb2hsv(img)
    for i, name in enumerate(['hue', 'saturation', 'value']):
        channel = hsv[..., i]
        features[f'{name}_mean'] = np.mean(channel)
        features[f'{name}_std'] = np.std(channel)
    
    # ======================
    # 3. COLOR HISTOGRAMS
    # ======================
    for i, name in enumerate(['red', 'green', 'blue']):
        hist = np.histogram(img[..., i], bins=bins, range=(0, 256))[0]
        hist = hist / hist.sum()  # Normalize
        for bin_idx in range(bins):
            features[f'{name}_hist_bin{bin_idx}'] = hist[bin_idx]
    
    # ======================
    # 4. TEXTURE (GLCM)
    # ======================
    gray = color.rgb2gray(img) * 255
    gray = gray.astype(np.uint8)
    glcm = graycomatrix(gray, distances=distances, angles=angles, levels=256)
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']:
        features[f'glcm_{prop}'] = graycoprops(glcm, prop)[0, 0]
    
    # ======================
    # 5. VEGETATION INDICES
    # ======================
    # RGB-based approximations (no NIR available)
    features['vari'] = np.nan_to_num((G - R) / (G + R - B + 1e-10))  # Visible Atmospherically Resistant Index
    features['exg'] = 2*G - R - B  # Excess Green Index
    features['cive'] = 0.441*R - 0.811*G + 0.385*B + 18.787  # Color Index of Vegetation Extraction

    # ===================
    # Convert Features to dataframe
    # ===================

    feature_df = pd.DataFrame([features])

    array_columns = ['vari', 'exg', 'cive']

    feature_df = summarize_array_column(feature_df, array_columns)
    
    return feature_df


# Streamlit app
def main():
    st.title("🌍 EuroSAT Image Classifier")
    
    # Load models
    models = load_models()
    
    # Image selection
    image_dir = "sample_images"
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_image = st.selectbox("Select Test Image", image_files)
    img_path = os.path.join(image_dir, selected_image)
    
    # Display image
    st.image(Image.open(img_path), width=300)
    
    # Feature extraction button
    if st.button("Extract Features"):
        with st.spinner("Extracting features..."):
            features_df = extract_advanced_features(img_path)
            st.session_state.features = features_df.iloc[0].values  # Store array for prediction
            st.success("Feature extraction complete!")
            
            # Display features
            st.subheader("Extracted Features")
            st.dataframe(features_df)
            
     # Prediction section
    if 'features' in st.session_state:
        st.divider()
        model_choice = st.radio("Select Model", ('SVM', 'Random Forest'))
        
        if st.button("Predict"):
            if model_choice == 'SVM':
                scaled_features = models['scaler'].transform([st.session_state.features])
                pred = models['svm'].predict(scaled_features)[0]
                proba = models['svm'].predict_proba(scaled_features)[0]
            else:
                pred = models['rf'].predict([st.session_state.features])[0]
                proba = models['rf'].predict_proba([st.session_state.features])[0]
            

            # Get class name from JSON mapping
            class_name = models['class_labels'].get(str(pred), f"Class {pred}")
            
            # Display results
            st.success(f"Predicted: {class_name}")
            
            # Show probabilities with class names
            proba_df = pd.DataFrame({
                'Class': [models['class_labels'].get(str(i), f"Class {i}") for i in range(len(proba))],
                'Probability': proba
            }).set_index('Class')
            
            st.bar_chart(proba_df)
            st.table(proba_df)


if __name__ == "__main__":
    main()
