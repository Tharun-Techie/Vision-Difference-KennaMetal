import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF
import tempfile

# Define folder paths
APPROVED_FOLDER = "Approved Images"
CORRUPT_FOLDER = "Corrupt Images"
OUTPUT_FOLDER = "Output Images"
REPORT_FOLDER = "Report Folder"

# Create output folders if they don't exist
for folder in [OUTPUT_FOLDER, REPORT_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def load_image(image_file):
    """Load an image using PIL and convert it to OpenCV format."""
    image = Image.open(image_file)
    image = np.array(image)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image

def compare_images(original, corrupt):
    """Compare images, highlight differences, and calculate change percentages."""
    if original.shape != corrupt.shape:
        return None, "Image dimensions do not match!"
    
    # Convert to grayscale
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
    
    # Compute absolute difference
    diff = cv2.absdiff(gray_original, gray_corrupt)
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    # Calculate % difference
    total_pixels = diff.size
    changed_pixels = np.count_nonzero(threshold_diff)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Highlight differences in red
    highlight = original.copy()
    highlight[threshold_diff > 0] = [255, 0, 0]
    
    return highlight, f"Changes Detected: {change_percentage:.2f}%"

def generate_pdf(original, corrupt, output, report_path, change_percentage):
    """Generate a PDF report for the image comparison."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Image Comparison Report", ln=True, align='C')
    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Change Percentage: {change_percentage:.2f}%", ln=True, align='C')
    
    # Save temporary images
    temp_original = "original_temp.jpg"
    temp_corrupt = "corrupt_temp.jpg"
    temp_output = "output_temp.jpg"
    cv2.imwrite(temp_original, original)
    cv2.imwrite(temp_corrupt, corrupt)
    cv2.imwrite(temp_output, output)
    
    # Add images to PDF
    pdf.image(temp_original, x=10, w=60)
    pdf.image(temp_corrupt, x=80, w=60)
    pdf.image(temp_output, x=150, w=60)
    
    # Save report
    pdf.output(report_path)
    os.remove(temp_original)
    os.remove(temp_corrupt)
    os.remove(temp_output)

def batch_process():
    """Process all images in batch mode."""
    for filename in os.listdir(APPROVED_FOLDER):
        original_path = os.path.join(APPROVED_FOLDER, filename)
        corrupt_path = os.path.join(CORRUPT_FOLDER, filename)
        output_path = os.path.join(OUTPUT_FOLDER, filename)
        report_path = os.path.join(REPORT_FOLDER, f"{os.path.splitext(filename)[0]}.pdf")
        
        if os.path.exists(corrupt_path):
            original = cv2.imread(original_path)
            corrupt = cv2.imread(corrupt_path)
            output, change_percentage = compare_images(original, corrupt)
            cv2.imwrite(output_path, output)
            generate_pdf(original, corrupt, output, report_path, float(change_percentage.split()[2][:-1]))
    st.success("Batch Processing Complete! Reports saved in Report Folder.")

# Streamlit UI
st.title("🔍 Image Comparison Tool")
mode = st.sidebar.radio("Select Mode", ("Single Comparison", "Batch Processing"))

if mode == "Single Comparison":
    st.sidebar.header("Upload Images")
    approved_image = st.sidebar.file_uploader("Upload Approved Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    
    if approved_image and corrupt_image:
        original = load_image(approved_image)
        corrupt = load_image(corrupt_image)
        output, change_percentage = compare_images(original, corrupt)
        
        if output is not None:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(original, caption="✅ Approved Image")
            with col2:
                st.image(corrupt, caption="⚠️ Corrupt Image")
            with col3:
                st.image(output, caption=f"🔴 Differences ({change_percentage})")
            
            # Generate report button
            if st.button("📄 Generate Report"):
                report_path = os.path.join(REPORT_FOLDER, "single_comparison_report.pdf")
                generate_pdf(original, corrupt, output, report_path, float(change_percentage.split()[2][:-1]))
                st.success("Report Generated!")
                with open(report_path, "rb") as file:
                    st.download_button("📥 Download Report", file, file_name="comparison_report.pdf", mime="application/pdf")

elif mode == "Batch Processing":
    if st.button("🚀 Start Batch Processing"):
        batch_process()
        
    # Show report folder
    st.subheader("📂 Available Reports")
    reports = os.listdir(REPORT_FOLDER)
    if reports:
        for report in reports:
            report_path = os.path.join(REPORT_FOLDER, report)
            with open(report_path, "rb") as file:
                st.download_button(f"📥 {report}", file, file_name=report, mime="application/pdf")
    else:
        st.info("No reports available.")
