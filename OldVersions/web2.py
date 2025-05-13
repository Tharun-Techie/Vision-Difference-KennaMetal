import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Streamlit App Title
st.title("🔍 TIFF Image Difference Finder")

# Upload two images: Approved & Corrupt
st.sidebar.header("Upload Images")
approved_image = st.sidebar.file_uploader("Upload Approved Image (TIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])
corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image (TIFF, PNG, JPG)", type=["tif", "tiff", "png", "jpg", "jpeg"])

def load_image(image_file):
    """Load an image using PIL and convert it to an OpenCV format."""
    try:
        image = Image.open(image_file)  # Load using PIL
        image = np.array(image)  # Convert to NumPy array
        if len(image.shape) == 2:  # Convert grayscale to 3-channel
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def compare_images(original, corrupt):
    """Compare two images and highlight differences in red."""
    try:
        # Ensure images have the same dimensions
        if original.shape != corrupt.shape:
            st.error("Images must have the same dimensions. Please upload matching images.")
            return None

        # Convert images to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Highlight differences in red
        highlight = original.copy()
        highlight[threshold_diff > 0] = [0, 0, 255]  # Red color for differences

        return highlight

    except Exception as e:
        st.error(f"Error comparing images: {e}")
        return None

# Process images when both are uploaded
if approved_image and corrupt_image:
    st.sidebar.success("✅ Images uploaded successfully!")

    # Load Images
    original = load_image(approved_image)
    corrupt = load_image(corrupt_image)

    if original is not None and corrupt is not None:
        # Compare Images
        output = compare_images(original, corrupt)

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
            with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_output:
                output_image = Image.fromarray(output)  # Convert to PIL format
                output_image.save(temp_output.name, format="TIFF")  # Save as TIFF
                st.sidebar.download_button(
                    label="📥 Download Output Image",
                    data=open(temp_output.name, "rb").read(),
                    file_name="image_difference.tif",
                    mime="image/tiff",
                )
                os.unlink(temp_output.name)  # Clean up

else:
    st.warning("📌 Please upload both Approved and Corrupt images to continue.")
