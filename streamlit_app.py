import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import io
import pandas as pd
import zipfile
import os
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Signature Verification AI",
    page_icon="✍️",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    try:
        # Try to load the model with safe_mode=False for Lambda layers
        model = tf.keras.models.load_model('siamese_model.keras', safe_mode=False)
        return model
    except Exception as e:
        st.error(f"""
        **Model Loading Error**: Could not load the trained model.
        
        **Possible solutions:**
        1. Ensure `siamese_model.keras` is in the project directory
        2. Check if the model file is corrupted
        3. Verify TensorFlow compatibility
        
        **Error details**: {str(e)}
        
        **Demo Mode**: You can still explore the interface, but predictions won't work.
        """)
        return None

def create_demo_model():
    """Create a simple demo model for testing the interface"""
    try:
        # Create a simple model with the same architecture for demo
        input_shape = (100, 100, 1)
        
        # Base network
        base_input = tf.keras.Input(shape=input_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(base_input)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D((2, 2))(x)
        x = tf.keras.layers.Flatten()(x)
        base_output = tf.keras.layers.Dense(128, activation='relu')(x)
        base_model = tf.keras.Model(base_input, base_output)
        
        # Siamese network
        input_a = tf.keras.Input(shape=input_shape)
        input_b = tf.keras.Input(shape=input_shape)
        feat_a = base_model(input_a)
        feat_b = base_model(input_b)
        distance = tf.keras.layers.Lambda(lambda tensors: tf.abs(tensors[0] - tensors[1]))([feat_a, feat_b])
        output = tf.keras.layers.Dense(1, activation='sigmoid')(distance)
        
        demo_model = tf.keras.Model([input_a, input_b], output)
        demo_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        return demo_model
    except:
        return None

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
        # Return a random similarity score for demo purposes
        import random
        similarity_score = random.uniform(0.2, 0.9)
        st.warning("⚠️ **Demo Mode**: Using random similarity scores. Upload the trained model for real predictions.")
    else:
        # Make real prediction
        similarity_score = model.predict([img1_batch, img2_batch], verbose=0)[0][0]
    
    return similarity_score

def batch_process_signatures(model, reference_image, comparison_images):
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
    
    # Load model
    model = load_model()
    demo_mode = model is None
    
    if demo_mode:
        st.warning("""
        🚧 **Demo Mode Active** 
        
        The trained model couldn't be loaded, but you can still explore the interface.
        Upload `siamese_model.keras` to enable real predictions.
        """)
        model = create_demo_model()  # Create a demo model for interface testing
    
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
        value=0.5, 
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
                        results_df = batch_process_signatures(model, reference_image, comparison_images)
                        
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
    main()