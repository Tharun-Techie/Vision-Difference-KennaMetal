import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Streamlit App Title
st.title("🔍 Image Difference Finder")

# Upload two images: Approved & Corrupt
st.sidebar.header("Upload Images")
approved_image = st.sidebar.file_uploader("Upload Approved Image", type=["png", "jpg", "jpeg"])
corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image", type=["png", "jpg", "jpeg"])

def compare_images(original, corrupt):
    """Compare two images and highlight differences in red."""
    try:
        # Convert uploaded images to OpenCV format
        original = cv2.imdecode(np.frombuffer(original.read(), np.uint8), cv2.IMREAD_COLOR)
        corrupt = cv2.imdecode(np.frombuffer(corrupt.read(), np.uint8), cv2.IMREAD_COLOR)

        # Check if images are loaded properly
        if original is None or corrupt is None:
            st.error("Error loading images. Please upload valid image files.")
            return None

        # Ensure both images have the same shape
        if original.shape != corrupt.shape:
            st.error("Images must have the same dimensions. Please upload matching images.")
            return None

        # Convert to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Highlight differences in red
        highlight = original.copy()
        highlight[threshold_diff > 0] = [0, 0, 255]  # Red color for differences

        return original, corrupt, highlight

    except Exception as e:
        st.error(f"Error comparing images: {e}")
        return None

# Process images when both are uploaded
if approved_image and corrupt_image:
    st.sidebar.success("✅ Images uploaded successfully!")

    # Compare Images
    original, corrupt, output = compare_images(approved_image, corrupt_image)

    if output is not None:
        # Display images in Streamlit
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(original, caption="✅ Approved Image", use_column_width=True)
        with col2:
            st.image(corrupt, caption="⚠️ Corrupt Image", use_column_width=True)
        with col3:
            st.image(output, caption="🔴 Differences Highlighted", use_column_width=True)

        # Save output image for download
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_output:
            cv2.imwrite(temp_output.name, output)
            st.sidebar.download_button(
                label="📥 Download Output Image",
                data=open(temp_output.name, "rb").read(),
                file_name="image_difference.jpg",
                mime="image/jpeg",
            )
            os.unlink(temp_output.name)  # Clean up

else:
    st.warning("📌 Please upload both Approved and Corrupt images to continue.")
