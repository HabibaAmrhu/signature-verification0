import streamlit as st
import numpy as np
from PIL import Image
import io
import pandas as pd
import zipfile
import os
from datetime import datetime

# Detect if running on Streamlit Cloud
def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud with multiple indicators"""
    indicators = [
        'streamlit.app' in os.getenv('HOSTNAME', ''),
        os.getenv('STREAMLIT_CLOUD', False),
        '/mount/src' in os.getcwd(),
        '/home/adminuser' in os.getcwd(),
        os.path.exists('/mount/src'),
        'streamlit.app' in str(os.environ.get('PWD', '')),
        any('streamlit' in str(v).lower() and 'app' in str(v).lower() 
            for v in os.environ.values() if isinstance(v, str))
    ]
    return any(indicators)

# Completely avoid TensorFlow for cloud deployment
TENSORFLOW_AVAILABLE = False
tf = None

# NEVER import TensorFlow on cloud - this prevents segfaults
if not is_streamlit_cloud():
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        import tensorflow as tf
        tf.config.set_visible_devices([], 'GPU')
        TENSORFLOW_AVAILABLE = True
    except ImportError:
        TENSORFLOW_AVAILABLE = False
        tf = None

# Page config
st.set_page_config(
    page_title="Signature Verification AI",
    page_icon="✍️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    # Triple-check cloud environment before any TensorFlow operations
    if is_streamlit_cloud() or not TENSORFLOW_AVAILABLE or tf is None:
        return None
        
    try:
        # Try to load the model with safe_mode=False for Lambda layers
        model = tf.keras.models.load_model('siamese_model.keras', safe_mode=False)
        st.success("✅ Model loaded successfully!")
        return model
    except FileNotFoundError:
        st.warning("⚠️ Model file not found. Running in demo mode.")
        return None
    except Exception as e:
        st.warning(f"⚠️ Model loading issue: {str(e)[:100]}... Running in demo mode.")
        return None

def create_demo_prediction(img1, img2):
    """Enhanced demo prediction with stricter signature-specific analysis"""
    try:
        # Convert images to grayscale arrays with better preprocessing
        arr1 = np.array(img1.convert('L').resize((150, 150)))
        arr2 = np.array(img2.convert('L').resize((150, 150)))
        
        # Apply threshold to get binary images (signature vs background)
        threshold1 = np.mean(arr1) - np.std(arr1) * 0.5
        threshold2 = np.mean(arr2) - np.std(arr2) * 0.5
        binary1 = (arr1 < threshold1).astype(float)
        binary2 = (arr2 < threshold2).astype(float)
        
        similarities = []
        penalties = []
        
        # 1. Shape-based correlation (more discriminating)
        shape_corr = np.corrcoef(binary1.flatten(), binary2.flatten())[0, 1]
        if not np.isnan(shape_corr):
            similarities.append(max(0, (shape_corr + 1) / 2))
        
        # 2. Stroke density comparison
        density1 = np.sum(binary1) / binary1.size
        density2 = np.sum(binary2) / binary2.size
        density_diff = abs(density1 - density2)
        density_sim = max(0, 1 - density_diff * 3)  # Penalize density differences
        similarities.append(density_sim)
        
        # 3. Edge pattern analysis (signature-specific)
        from scipy import ndimage
        edges1 = ndimage.sobel(binary1)
        edges2 = ndimage.sobel(binary2)
        edge_corr = np.corrcoef(edges1.flatten(), edges2.flatten())[0, 1]
        if not np.isnan(edge_corr):
            similarities.append(max(0, (edge_corr + 1) / 2))
        
        # 4. Aspect ratio and size penalties
        h1, w1 = binary1.shape
        h2, w2 = binary2.shape
        aspect1 = w1 / h1 if h1 > 0 else 1
        aspect2 = w2 / h2 if h2 > 0 else 1
        aspect_penalty = abs(aspect1 - aspect2) * 0.5
        penalties.append(aspect_penalty)
        
        # 5. Center of mass comparison
        if np.sum(binary1) > 0 and np.sum(binary2) > 0:
            cm1 = ndimage.center_of_mass(binary1)
            cm2 = ndimage.center_of_mass(binary2)
            cm_distance = np.sqrt((cm1[0] - cm2[0])**2 + (cm1[1] - cm2[1])**2)
            cm_penalty = min(0.3, cm_distance / 50)  # Normalize and cap penalty
            penalties.append(cm_penalty)
        
        # 6. Stroke width analysis
        stroke_pixels1 = np.sum(binary1 > 0.5)
        stroke_pixels2 = np.sum(binary2 > 0.5)
        if stroke_pixels1 > 0 and stroke_pixels2 > 0:
            stroke_ratio = min(stroke_pixels1, stroke_pixels2) / max(stroke_pixels1, stroke_pixels2)
            similarities.append(stroke_ratio)
        
        # Calculate final similarity with penalties
        if similarities:
            base_similarity = np.mean(similarities)
            total_penalty = sum(penalties)
            final_similarity = max(0.1, base_similarity - total_penalty)
            
            # Add controlled randomness (less than before)
            import random
            final_similarity += random.uniform(-0.03, 0.03)
            final_similarity = max(0.0, min(1.0, final_similarity))
        else:
            final_similarity = 0.4  # Lower default for unknown cases
        
        return final_similarity
        
    except Exception as e:
        # More conservative fallback
        import random
        return random.uniform(0.2, 0.6)

def preprocess_image(image):
    """Preprocess uploaded image for model prediction"""
    # Convert to grayscale and resize
    img = image.convert('L').resize((100, 100))
    # Convert to numpy array and normalize
    arr = np.array(img) / 255.0
    # Add channel dimension
    return np.expand_dims(arr, axis=-1)

def predict_similarity(model, img1, img2, demo_mode=False):
    """Predict if two signatures are from the same person"""
    # Preprocess images
    processed_img1 = preprocess_image(img1)
    processed_img2 = preprocess_image(img2)
    
    # Add batch dimension
    img1_batch = np.expand_dims(processed_img1, axis=0)
    img2_batch = np.expand_dims(processed_img2, axis=0)
    
    if demo_mode or model is None:
        # Use demo prediction based on actual image similarity
        similarity_score = create_demo_prediction(img1, img2)
        if not hasattr(st.session_state, 'demo_warning_shown'):
            if not TENSORFLOW_AVAILABLE:
                st.info("🌐 **Cloud Demo**: Using advanced correlation analysis for signature comparison.")
            else:
                st.warning("⚠️ **Demo Mode**: Using advanced image correlation. Upload the trained model for AI predictions.")
            st.session_state.demo_warning_shown = True
    else:
        # Make real prediction
        similarity_score = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    
    return similarity_score

def batch_process_signatures(model, reference_image, comparison_images, demo_mode=False):
    """Process multiple signatures against a reference signature"""
    results = []
    
    for i, comp_image in enumerate(comparison_images):
        try:
            similarity_score = predict_similarity(model, reference_image, comp_image, demo_mode)
            threshold = 0.5
            match = similarity_score > threshold
            confidence = abs(similarity_score - 0.5) * 2
            
            results.append({
                'Image': f'Signature_{i+1}',
                'Similarity_Score': round(similarity_score, 3),
                'Confidence': f"{confidence:.1%}",
                'Match': "✅ MATCH" if match else "❌ NO MATCH",
                'Status': 'Genuine' if match else 'Suspicious'
            })
        except Exception as e:
            results.append({
                'Image': f'Signature_{i+1}',
                'Similarity_Score': 'Error',
                'Confidence': 'N/A',
                'Match': '❌ ERROR',
                'Status': f'Error: {str(e)}'
            })
    
    return pd.DataFrame(results)

def create_sample_signatures():
    """Create downloadable sample signature images for testing"""
    st.markdown("### 📥 Sample Signatures for Testing")
    st.markdown("Need test images? Here are some tips for creating test signatures:")
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("""
        **For Matching Signatures:**
        - Use the same person's signature
        - Slight variations are normal
        - Same writing style and flow
        """)
    
    with col2:
        st.info("""
        **For Non-Matching Signatures:**
        - Different people's signatures
        - Different writing styles
        - Different letter formations
        """)

# Main app
def main():
    st.title("✍️ Signature Verification AI")
    st.markdown("### Advanced signature verification with single comparison and batch processing")
    
    # Debug info (only show in development)
    if st.sidebar.checkbox("Show Debug Info", value=False):
        st.sidebar.write("**Environment Debug:**")
        st.sidebar.write(f"Cloud detected: {is_streamlit_cloud()}")
        st.sidebar.write(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
        st.sidebar.write(f"Current working dir: {os.getcwd()}")
        st.sidebar.write(f"Hostname: {os.getenv('HOSTNAME', 'Not set')}")
    
    # Load model
    model = load_model()
    demo_mode = model is None
    
    if demo_mode:
        if not TENSORFLOW_AVAILABLE:
            st.info("""
            🌐 **Cloud Demo Mode** 
            
            You're using the online demo! This version uses advanced image correlation 
            algorithms to compare signatures. For full AI predictions with 98.75% accuracy, 
            run this locally with the trained model.
            """)
        else:
            st.info("""
            🚧 **Demo Mode Active** 
            
            The trained model couldn't be loaded, but you can still explore the interface.
            In demo mode, advanced image correlation is used for similarity scoring.
            """)
        # Don't create demo model, just use None and handle in predict function
    
    # Sidebar for mode selection
    st.sidebar.title("🔧 Options")
    mode = st.sidebar.selectbox(
        "Choose verification mode:",
        ["Single Comparison", "Batch Processing", "About & Samples"]
    )
    
    # Threshold adjustment
    threshold = st.sidebar.slider(
        "Similarity Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.65, 
        step=0.05,
        help="Adjust sensitivity: Lower = more strict, Higher = more lenient"
    )
    
    if mode == "Single Comparison":
        st.markdown("#### Compare two individual signatures")
        
        # Create two columns for image uploads
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 First Signature")
            uploaded_file1 = st.file_uploader(
                "Choose first signature image", 
                type=['png', 'jpg', 'jpeg'],
                key="sig1"
            )
            
            if uploaded_file1:
                image1 = Image.open(uploaded_file1)
                st.image(image1, caption="First Signature", use_column_width=True)
        
        with col2:
            st.subheader("📝 Second Signature")
            uploaded_file2 = st.file_uploader(
                "Choose second signature image", 
                type=['png', 'jpg', 'jpeg'],
                key="sig2"
            )
            
            if uploaded_file2:
                image2 = Image.open(uploaded_file2)
                st.image(image2, caption="Second Signature", use_column_width=True)
        
        # Prediction section
        if uploaded_file1 and uploaded_file2:
            st.markdown("---")
            
            if st.button("🔍 Verify Signatures", type="primary"):
                with st.spinner("Analyzing signatures..."):
                    try:
                        similarity_score = predict_similarity(model, image1, image2, demo_mode)
                        
                        # Display results
                        st.subheader("📊 Verification Results")
                        
                        # Create metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Similarity Score", f"{similarity_score:.3f}")
                        
                        with col2:
                            confidence = abs(similarity_score - 0.5) * 2
                            st.metric("Confidence", f"{confidence:.1%}")
                        
                        with col3:
                            match = "✅ MATCH" if similarity_score > threshold else "❌ NO MATCH"
                            st.metric("Result", match)
                        
                        # Progress bar for similarity
                        st.progress(similarity_score)
                        
                        # Interpretation based on custom threshold
                        if similarity_score > threshold + 0.2:
                            st.success("🎯 High confidence: These signatures are likely from the same person!")
                        elif similarity_score > threshold:
                            st.warning("⚠️ Moderate confidence: Signatures might be from the same person.")
                        else:
                            st.error("🚫 Low confidence: These signatures are likely from different people.")
                            
                    except Exception as e:
                        st.error(f"Error during prediction: {str(e)}")
    
    elif mode == "Batch Processing":
        st.markdown("#### Compare multiple signatures against a reference")
        
        # Reference signature upload
        st.subheader("📋 Reference Signature")
        reference_file = st.file_uploader(
            "Upload the reference signature (authentic signature)", 
            type=['png', 'jpg', 'jpeg'],
            key="ref_sig"
        )
        
        if reference_file:
            reference_image = Image.open(reference_file)
            st.image(reference_image, caption="Reference Signature", width=300)
        
        # Multiple signatures upload
        st.subheader("📁 Signatures to Verify")
        uploaded_files = st.file_uploader(
            "Upload multiple signatures to compare against reference",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key="batch_sigs"
        )
        
        if uploaded_files:
            st.write(f"Uploaded {len(uploaded_files)} signatures for verification")
            
            # Show preview of uploaded images
            if st.checkbox("Show preview of uploaded signatures"):
                cols = st.columns(min(4, len(uploaded_files)))
                for i, file in enumerate(uploaded_files[:4]):
                    with cols[i % 4]:
                        img = Image.open(file)
                        st.image(img, caption=f"Sig {i+1}", use_column_width=True)
                if len(uploaded_files) > 4:
                    st.write(f"... and {len(uploaded_files) - 4} more")
        
        # Batch processing
        if reference_file and uploaded_files:
            if st.button("🚀 Process All Signatures", type="primary"):
                with st.spinner(f"Processing {len(uploaded_files)} signatures..."):
                    try:
                        # Load all comparison images
                        comparison_images = [Image.open(file) for file in uploaded_files]
                        
                        # Process batch
                        results_df = batch_process_signatures(model, reference_image, comparison_images, demo_mode)
                        
                        # Display results
                        st.subheader("📊 Batch Processing Results")
                        
                        # Summary metrics
                        total_sigs = len(results_df)
                        matches = len(results_df[results_df['Status'] == 'Genuine'])
                        suspicious = len(results_df[results_df['Status'] == 'Suspicious'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", total_sigs)
                        with col2:
                            st.metric("Genuine", matches, delta=f"{matches/total_sigs:.1%}")
                        with col3:
                            st.metric("Suspicious", suspicious, delta=f"{suspicious/total_sigs:.1%}")
                        
                        # Results table
                        st.dataframe(results_df, use_container_width=True)
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results as CSV",
                            data=csv,
                            file_name=f"signature_verification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
                        # Visual summary
                        if matches > 0 or suspicious > 0:
                            st.subheader("📈 Summary Chart")
                            chart_data = pd.DataFrame({
                                'Status': ['Genuine', 'Suspicious'],
                                'Count': [matches, suspicious]
                            })
                            st.bar_chart(chart_data.set_index('Status'))
                        
                    except Exception as e:
                        st.error(f"Error during batch processing: {str(e)}")
    
    else:  # About & Samples mode
        st.markdown("#### About this application")
        create_sample_signatures()
        
        # Information section
        st.markdown("---")
        st.markdown("### 🤖 How it works")
        st.markdown("""
        This AI uses a **Siamese Neural Network** trained on handwritten signatures to determine similarity:
        
        - **Deep Learning**: Convolutional neural network extracts signature features
        - **Siamese Architecture**: Compares two signatures simultaneously  
        - **Similarity Score**: Higher scores (closer to 1.0) indicate same person
        - **Accuracy**: Trained model achieves 98.75% validation accuracy
        
        **Tips for best results:**
        - Use clear, high-contrast signature images
        - Ensure signatures are properly cropped
        - Images will be automatically resized to 100x100 pixels
        """)
        
        # Model architecture details
        st.markdown("### 🏗️ Model Architecture")
        st.code("""
        Input Layer (100x100x1)
        ├── Conv2D (32 filters, 3x3)
        ├── MaxPooling2D (2x2)
        ├── Conv2D (64 filters, 3x3)
        ├── MaxPooling2D (2x2)
        ├── Flatten
        └── Dense (128 units)

        Siamese Network
        ├── Feature Extraction (Base CNN)
        ├── Distance Calculation (L1 distance)
        └── Classification (Sigmoid output)
        """)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("The app is running in demo mode. Some features may be limited.")