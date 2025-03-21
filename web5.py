import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF
import tempfile

# Define folder paths
approved_folder = "Approved Images"
corrupt_folder = "Corrupt Images"
output_folder = "Output Images"
report_folder = "Report Folder"

# Create necessary folders if they don't exist
for folder in [approved_folder, corrupt_folder, output_folder, report_folder]:
    os.makedirs(folder, exist_ok=True)

def load_image(image_file):
    try:
        image = Image.open(image_file)
        image = np.array(image)
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

def compare_images(original, corrupt):
    if original.shape != corrupt.shape:
        st.error("Images must have the same dimensions. Please upload matching images.")
        return None

    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_original, gray_corrupt)
    _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
    highlight = original.copy()
    highlight[threshold_diff > 0] = [255, 0, 0]  # Highlight differences in red
    return highlight

def generate_pdf_report(original_path, corrupt_path, output_path, report_path):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Comparison Report: {os.path.basename(original_path)}", ln=True, align='C')
    pdf.ln(8)

    original = cv2.imread(original_path)
    corrupt = cv2.imread(corrupt_path)
    highlight = cv2.imread(output_path)

    if original is None or corrupt is None:
        return

    temp_original = "temp_original.jpg"
    temp_corrupt = "temp_corrupt.jpg"
    temp_output = "temp_output.jpg"
    cv2.imwrite(temp_original, original)
    cv2.imwrite(temp_corrupt, corrupt)
    cv2.imwrite(temp_output, highlight)

    img_width = 45
    pdf.cell(45, 10, txt="Approved Image", ln=True, align='C')
    pdf.image(temp_original, x=10, y=pdf.get_y(), w=img_width)
    pdf.ln(70)

    pdf.cell(45, 10, txt="Corrupt Image", ln=True, align='C')
    pdf.image(temp_corrupt, x=10, y=pdf.get_y(), w=img_width)
    pdf.ln(70)

    pdf.cell(95, 10, txt="Output Image (Differences Highlighted)", ln=True, align='C')
    pdf.image(temp_output, x=10, y=pdf.get_y(), w=img_width)
    pdf.ln(70)

    pdf.output(report_path, "F")
    os.remove(temp_original)
    os.remove(temp_corrupt)
    os.remove(temp_output)

def batch_process_images():
    for filename in os.listdir(approved_folder):
        if filename.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".bmp")):
            original_path = os.path.join(approved_folder, filename)
            corrupt_path = os.path.join(corrupt_folder, filename)
            output_path = os.path.join(output_folder, filename)
            report_path = os.path.join(report_folder, f"{os.path.splitext(filename)[0]}.pdf")
            if os.path.exists(corrupt_path):
                original = cv2.imread(original_path)
                corrupt = cv2.imread(corrupt_path)
                if original is not None and corrupt is not None:
                    output = compare_images(original, corrupt)
                    if output is not None:
                        cv2.imwrite(output_path, output)
                        generate_pdf_report(original_path, corrupt_path, output_path, report_path)

def main():
    st.title("🔍 TIFF Image Difference Finder & Batch Processor")
    mode = st.sidebar.radio("Choose Mode:", ("📂 Batch Processing", "📤 Upload & Compare"))

    if mode == "📂 Batch Processing":
        if st.sidebar.button("Run Batch Processing"):
            batch_process_images()
            st.success("✅ Batch processing completed! Check 'Report Folder'.")

    elif mode == "📤 Upload & Compare":
        st.sidebar.header("Upload Images")
        approved_image = st.sidebar.file_uploader("Upload Approved Image", type=["tif", "tiff", "png", "jpg", "jpeg"])
        corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image", type=["tif", "tiff", "png", "jpg", "jpeg"])

        if approved_image and corrupt_image:
            original = load_image(approved_image)
            corrupt = load_image(corrupt_image)
            if original is not None and corrupt is not None:
                output = compare_images(original, corrupt)
                if output is not None:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.image(original, caption="✅ Approved Image", use_container_width=True)
                    with col2:
                        st.image(corrupt, caption="⚠️ Corrupt Image", use_container_width=True)
                    with col3:
                        st.image(output, caption="🔴 Differences Highlighted", use_container_width=True)

                    with tempfile.NamedTemporaryFile(delete=False, suffix=".tif") as temp_output:
                        output_image = Image.fromarray(output)
                        output_image.save(temp_output.name, format="TIFF")
                    with open(temp_output.name, "rb") as file:
                        st.sidebar.download_button("📥 Download Output Image", data=file, file_name="image_difference.tif", mime="image/tiff")
                    os.remove(temp_output.name)

if __name__ == "__main__":
    main()
