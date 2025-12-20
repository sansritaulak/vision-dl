import streamlit as st
from inference import predict
from PIL import Image
import os

st.set_page_config(page_title="CIFAR-10 Real-World Classifier", layout="centered")
st.title("CIFAR-10 Image Classifier")
st.write("Upload any photo — dog, cat, airplane, car — and see what my model thinks!")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Your uploaded image", use_container_width=True)
    
    # Save temporarily
    temp_path = "temp_uploaded.jpg"
    image.save(temp_path)
    
    with st.spinner("Running model..."):
        try:
            result = predict(temp_path)
            
            if result:
                st.success(f"**Prediction: {result['prediction'].upper()}**")
                st.info(f"**Confidence: {result['confidence']:.1%}**")
            else:
                st.error("Prediction failed!")
                
        except Exception as e:
            st.error(f"Error: {e}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
else:
    st.info("Waiting for your image...")

st.caption("Trained on CIFAR-10 • ResNet18 • Temperature calibrated • Works on real-world photos!")