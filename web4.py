import streamlit as st
import cv2
import numpy as np
import os
from datetime import datetime

# Function to compare images and generate report
def compare_images(original_path, corrupt_path, output_path):
    try:
        original = cv2.imread(original_path)
        corrupt = cv2.imread(corrupt_path)

        if original is None or corrupt is None:
            st.error("❌ Error loading images. Please check the files.")
            return None, None

        if original.shape != corrupt.shape:
            st.warning("⚠️ Image sizes don't match. Please select correct images.")
            return None, None

        # Convert images to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Find contours to highlight differences
        contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes around differences
        highlighted_image = original.copy()
        cv2.drawContours(highlighted_image, contours, -1, (0, 255, 0), 2)  # Green contours

        # Calculate percentage difference
        total_pixels = diff.shape[0] * diff.shape[1]
        changed_pixels = np.sum(threshold_diff > 0)
        difference_percentage = (changed_pixels / total_pixels) * 100

        # Save output image
        cv2.imwrite(output_path, highlighted_image)

        # Generate report as a string
        report_content = (
            f"✅ Image Comparison Report\n"
            f"---------------------------------\n"
            f"🔹 Differences Found: {'Yes' if changed_pixels > 0 else 'No'}\n"
            f"🔹 Total Pixels: {total_pixels}\n"
            f"🔹 Changed Pixels: {changed_pixels}\n"
            f"🔹 Percentage Difference: {difference_percentage:.2f}%\n"
            f"🔹 Output Image Saved At: {output_path}\n"
        )

        return highlighted_image, report_content

    except Exception as e:
        st.error(f"❌ Error processing images: {e}")
        return None, None

# Streamlit UI
st.title("🔍 Image Difference Detector")

# Upload images
approved_img = st.file_uploader("📤 Upload Approved Image (TIF)", type=["tif"])
corrupt_img = st.file_uploader("📤 Upload Corrupted Image (TIF)", type=["tif"])

if approved_img and corrupt_img:
    # Save uploaded files to temporary paths
    approved_path = f"temp_approved.tif"
    corrupt_path = f"temp_corrupt.tif"
    output_path = f"temp_output.tif"

    with open(approved_path, "wb") as f:
        f.write(approved_img.getbuffer())
    with open(corrupt_path, "wb") as f:
        f.write(corrupt_img.getbuffer())

    # Compare images
    if st.button("🔎 Compare Images"):
        highlighted_image, report_text = compare_images(approved_path, corrupt_path, output_path)

        if highlighted_image is not None:
            st.image(highlighted_image, caption="🔴 Differences Highlighted", use_container_width=True)

            # Save report with UTF-8 encoding
            report_filename = f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(report_filename, "w", encoding="utf-8") as report_file:
                report_file.write(report_text)


            # Show report and download button
            st.text_area("📄 Comparison Report", report_text, height=200)
            st.download_button(label="📥 Download Report", data=report_text, file_name=report_filename, mime="text/plain")

    # Clean up temporary files
    os.remove(approved_path)
    os.remove(corrupt_path)
