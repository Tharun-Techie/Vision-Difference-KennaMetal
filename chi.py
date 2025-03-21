import os
import cv2
import numpy as np
from fpdf import FPDF

# Define folder paths
approved_folder = "Approved Images"
corrupt_folder = "Corrupt Images"
output_folder = "Output Images"
report_folder = "Report Folder"

# Create output folders if they don't exist
for folder in [output_folder, report_folder]:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"Created folder: {folder}")
    else:
        print(f"Folder already exists: {folder}")

def compare_images(original_path, corrupt_path, report_path, output_path):
    try:
        original = cv2.imread(original_path)
        corrupt = cv2.imread(corrupt_path)

        if original is None or corrupt is None:
            print(f"Error loading images: {original_path} or {corrupt_path}")
            return

        if original.shape != corrupt.shape:
            print(f"Skipping {original_path} - Image sizes don't match")
            return

        # Convert images to grayscale
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)

        # Compute absolute difference
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

        # Check if there are differences
        has_difference = np.any(threshold_diff > 0)

        # Highlight differences in red
        highlight = original.copy()
        highlight[threshold_diff > 0] = [0, 0, 255]  # Red color for differences

        # Save output image with highlighted differences
        cv2.imwrite(output_path, highlight)

        # Generate PDF Report
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Arial", style='', size=12)
        pdf.cell(200, 10, txt=f"Comparison Report: {os.path.basename(original_path)}", ln=True, align='C')
        pdf.ln(8)

        if not has_difference:
            pdf.set_text_color(0, 128, 0)  # Green text
            pdf.cell(200, 10, txt="No changes detected. The images are identical.", ln=True, align='C')
            print(f"No changes detected: {original_path}")
        else:
            pdf.set_text_color(255, 0, 0)  # Red text
            pdf.cell(200, 10, txt="Differences detected!", ln=True, align='C')
            print(f"Differences detected: {original_path}")

        pdf.ln(10)
        pdf.set_text_color(0, 0, 0)  # Reset to black

        # Save temporary images for PDF
        temp_original = "temp_original.jpg"
        temp_corrupt = "temp_corrupt.jpg"
        temp_output = "temp_output.jpg"
        cv2.imwrite(temp_original, original)
        cv2.imwrite(temp_corrupt, corrupt)
        cv2.imwrite(temp_output, highlight)

        # Adjust image width to fit side by side on a single page
        img_width = 45  # Half-page width

        # Add images to the PDF
        pdf.cell(45, 10, txt="Approved Image", ln=True, align='C')
        pdf.image(temp_original, x=10, y=pdf.get_y(), w=img_width)
        pdf.ln(70)  # Adjust space

        pdf.cell(45, 10, txt="Corrupt Image", ln=True, align='C')
        pdf.image(temp_corrupt, x=10, y=pdf.get_y(), w=img_width)
        pdf.ln(70)  # Adjust space

        pdf.cell(95, 10, txt="Output Image (Changes Made In New Image)", ln=True, align='C')
        pdf.image(temp_output, x=10, y=pdf.get_y(), w=img_width)
        pdf.ln(70)  # Adjust space

        # Save PDF
        pdf.output(report_path, "F")

        # Cleanup temporary images
        os.remove(temp_original)
        os.remove(temp_corrupt)
        os.remove(temp_output)

        print(f"Report generated: {report_path}")

    except Exception as e:
        print(f"Error in image comparison: {e}")

# Process images
for filename in os.listdir(approved_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')):
        original_path = os.path.join(approved_folder, filename)
        corrupt_path = os.path.join(corrupt_folder, filename)
        output_path = os.path.join(output_folder, filename)
        report_path = os.path.join(report_folder, f"{os.path.splitext(filename)[0]}.pdf")

        print(f"Processing: {filename}")

        if os.path.exists(corrupt_path):
            compare_images(original_path, corrupt_path, report_path, output_path)
        else:
            print(f"No corresponding corrupt file found for {filename}")

    else:
        print(f"Skipping non-image file: {filename}")

print("\nProcessing complete! Check 'Report Folder'.")