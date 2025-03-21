import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from fpdf import FPDF

def create_folders(folders):
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

def compare_images(original_path, corrupt_path, output_path, report_path):
    try:
        original = cv2.imread(original_path)
        corrupt = cv2.imread(corrupt_path)

        if original is None or corrupt is None:
            print(f"Error loading images: {original_path} or {corrupt_path}")
            return

        if original.shape != corrupt.shape:
            print(f"Skipping {original_path} - Image sizes don't match")
            return

        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)

        score, diff = ssim(gray_original, gray_corrupt, full=True)
        diff = (diff * 255).astype(np.uint8)

        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        has_difference = np.any(threshold_diff > 0)

        highlight = original.copy()
        highlight[threshold_diff > 0] = [0, 0, 255]
        cv2.imwrite(output_path, highlight)

        generate_report(original, corrupt, highlight, report_path, has_difference, score)

    except Exception as e:
        print(f"Error in image comparison: {e}")

def generate_report(original, corrupt, highlight, report_path, has_difference, score):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", '', 12)
        pdf.cell(200, 10, txt="Image Comparison Report", ln=True, align='C')
        pdf.ln(8)

        if not has_difference:
            pdf.set_text_color(0, 128, 0)
            pdf.cell(200, 10, txt="No changes detected. Images are identical.", ln=True, align='C')
        else:
            pdf.set_text_color(255, 0, 0)
            pdf.cell(200, 10, txt=f"Differences detected! Similarity Score: {score:.2f}", ln=True, align='C')

        pdf.set_text_color(0, 0, 0)
        temp_files = ['temp_original.jpg', 'temp_corrupt.jpg', 'temp_output.jpg']

        cv2.imwrite(temp_files[0], original)
        cv2.imwrite(temp_files[1], corrupt)
        cv2.imwrite(temp_files[2], highlight)

        img_width = 60
        for i, title in enumerate(["Approved Image", "Corrupt Image", "Highlighted Differences"]):
            pdf.cell(200, 10, txt=title, ln=True, align='C')
            pdf.image(temp_files[i], x=10, y=pdf.get_y(), w=img_width)
            pdf.ln(70)

        pdf.output(report_path)

        for file in temp_files:
            if os.path.exists(file):
                os.remove(file)

    except Exception as e:
        print(f"Error generating report: {e}")

def process_images(approved_folder, corrupt_folder, output_folder, report_folder):
    try:
        create_folders([output_folder, report_folder])

        approved_files = set(f.lower() for f in os.listdir(approved_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))
        corrupt_files = set(f.lower() for f in os.listdir(corrupt_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')))

        for filename in approved_files:
            original_path = os.path.join(approved_folder, filename)
            corrupt_path = os.path.join(corrupt_folder, filename)
            output_path = os.path.join(output_folder, filename)
            report_path = os.path.join(report_folder, f"{os.path.splitext(filename)[0]}.pdf")

            if filename in corrupt_files:
                compare_images(original_path, corrupt_path, output_path, report_path)
            else:
                print(f"No corresponding corrupt file found for: {filename}")

    except Exception as e:
        print(f"Error processing images: {e}")
if __name__ == "__main__":
    process_images("Approved Images", "Corrupt Images", "Output Images", "Report Folder")
    print("Processing complete! Check 'Report Folder'.")