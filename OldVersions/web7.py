import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF

# Ensure necessary folders exist
def ensure_folders_exist():
    folders = ["Approved Images", "Corrupt Images", "Output Images", "Report Folder"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

ensure_folders_exist()

# Load image using PIL and convert to OpenCV format
def load_image(image_file):
    try:
        image = Image.open(image_file)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    except Exception as e:
        print(f"Error loading image: {image_file}. {e}")
        return None

# Compare images and highlight differences
def compare_images(original, corrupt):
    if original is None or corrupt is None:
        return None, "Error: One or both images could not be loaded!"
    
    if original.shape != corrupt.shape:
        return None, "Error: Image dimensions do not match!"
    
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
    
    diff = cv2.absdiff(gray_original, gray_corrupt)
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    
    total_pixels = diff.size
    changed_pixels = np.count_nonzero(threshold_diff)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    highlight = original.copy()
    highlight[threshold_diff > 0] = [255, 0, 0]  # Highlight changes in red
    
    return highlight, f"Changes Detected: {change_percentage:.2f}%"

# Generate a PDF report for image comparison
def generate_pdf(original, corrupt, output, report_path, change_percentage):
    if original is None or corrupt is None or output is None:
        print("Error: One or more images could not be processed for the report.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(200, 10, txt="Image Comparison Report", ln=True, align='C')
    pdf.ln(8)

    pdf.set_font("Arial", style='B', size=12)
    pdf.cell(200, 10, txt=f"Change Percentage: {change_percentage:.2f}%", ln=True, align='C')
    pdf.ln(10)

    if change_percentage == 0:
        pdf.set_text_color(0, 128, 0)  # Green text
        pdf.cell(200, 10, txt="No changes detected. The images are identical.", ln=True, align='C')
    else:
        pdf.set_text_color(255, 0, 0)  # Red text
        pdf.cell(200, 10, txt="Differences detected!", ln=True, align='C')

    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)

    # Save temporary images
    temp_files = {
        "Approved Image": "approved.jpg",
        "Corrupt Image": "corrupted.jpg",
        "Difference Highlighted": "difference.jpg"
    }

    cv2.imwrite(temp_files["Approved Image"], original)
    cv2.imwrite(temp_files["Corrupt Image"], corrupt)
    cv2.imwrite(temp_files["Difference Highlighted"], output)

    img_width = 60  

    for label, temp_file in temp_files.items():
        pdf.set_font("Arial", style='B', size=10)
        pdf.cell(200, 10, txt=label, ln=True, align='C')
        pdf.image(temp_file, x=(210 - img_width) / 2, w=img_width)
        pdf.ln(70)  

    for temp_file in temp_files.values():
        os.remove(temp_file)

    pdf.output(report_path, "F")
    print(f"PDF Report saved at {report_path}")

# Batch process images
def batch_process():
    for filename in os.listdir("Approved Images"):
        original_path = os.path.join("Approved Images", filename)
        corrupt_path = os.path.join("Corrupt Images", filename)
        output_path = os.path.join("Output Images", filename)
        report_path = os.path.join("Report Folder", f"{os.path.splitext(filename)[0]}.pdf")

        if not os.path.exists(corrupt_path):
            print(f"Skipping {filename}: Corrupt image not found.")
            continue

        original, corrupt = cv2.imread(original_path), cv2.imread(corrupt_path)

        # Check if images are loaded correctly
        if original is None or corrupt is None:
            print(f"Error: Could not read {filename}. Skipping...")
            continue

        output, change_percentage = compare_images(original, corrupt)
        if output is not None:
            cv2.imwrite(output_path, output)
            generate_pdf(original, corrupt, output, report_path, float(change_percentage.split()[2][:-1]))

    st.success("Batch Processing Complete! Reports saved in 'Report Folder'.")

# Streamlit UI
st.title("🔍 Image Comparison Tool")
mode = st.sidebar.radio("Select Mode", ("Single Comparison", "Batch Processing"))

if mode == "Single Comparison":
    st.sidebar.header("Upload Images")
    approved_image = st.sidebar.file_uploader("Upload Approved Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
    corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image", type=["png", "jpg", "jpeg", "tif", "tiff"])

    if approved_image and corrupt_image:
        original, corrupt = load_image(approved_image), load_image(corrupt_image)
        output, change_percentage = compare_images(original, corrupt)

        if output is not None:
            col1, col2, col3 = st.columns(3)
            col1.image(original, caption="✅ Approved Image")
            col2.image(corrupt, caption="⚠️ Corrupt Image")
            col3.image(output, caption=f"🔴 Differences ({change_percentage})")

            if st.button("📄 Generate Report"):
                report_path = os.path.join("Report Folder", "single_comparison_report.pdf")
                generate_pdf(original, corrupt, output, report_path, float(change_percentage.split()[2][:-1]))
                with open(report_path, "rb") as file:
                    st.download_button("📥 Download Report", file, file_name="comparison_report.pdf", mime="application/pdf")

elif mode == "Batch Processing":
    if st.button("🚀 Start Batch Processing"):
        batch_process()

    st.subheader("📂 Available Reports")
    reports = os.listdir("Report Folder")
    if reports:
        for report in reports:
            report_path = os.path.join("Report Folder", report)
            with open(report_path, "rb") as file:
                st.download_button(f"📥 {report}", file, file_name=report, mime="application/pdf")
    else:
        st.info("No reports available.")
