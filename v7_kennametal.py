import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from fpdf import FPDF
import shutil
from datetime import datetime
import threading
import time

# Set page configuration
st.set_page_config(
    page_title="3D Image Analysis Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure necessary folders exist
def ensure_folders_exist():
    folders = ["Approved Images", "Corrupt Images", "Output Images", "Report Folder", "Temp"]
    for folder in folders:
        os.makedirs(folder, exist_ok=True)

ensure_folders_exist()

# Load image using PIL and convert to OpenCV format
def load_image(image_file):
    try:
        # Handle both file upload and file path
        if isinstance(image_file, str):
            image = Image.open(image_file)
        else:
            image = Image.open(image_file)
            
        # Convert to numpy array
        image_np = np.array(image)
        
        # Handle different color modes
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif len(image_np.shape) == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        
        return image_np
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None

# Save uploaded file to directory
def save_uploaded_file(uploaded_file, directory):
    if uploaded_file is not None:
        file_path = os.path.join(directory, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    return None

# Compare images and highlight differences with enhanced functionality
def compare_images(original, corrupt, threshold_value=30):
    if original is None or corrupt is None:
        return None, None, "Error: One or both images could not be loaded!"
    
    # Resize images if dimensions don't match
    if original.shape != corrupt.shape:
        st.warning("Image dimensions do not match! Resizing corrupt image to match approved image.")
        corrupt = cv2.resize(corrupt, (original.shape[1], original.shape[0]))
    
    # Convert images to grayscale
    gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
    
    # Calculate absolute difference
    diff = cv2.absdiff(gray_original, gray_corrupt)
    
    # Apply threshold to highlight significant differences
    _, threshold_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
    
    # Calculate difference metrics
    total_pixels = diff.size
    changed_pixels = np.count_nonzero(threshold_diff)
    change_percentage = (changed_pixels / total_pixels) * 100
    
    # Create difference heatmap (for visualization)
    diff_heatmap = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
    
    # Create highlighted image showing changes
    highlight = original.copy()
    # Create a mask where differences are detected
    mask = threshold_diff > 0
    # Apply red highlighting to those areas
    highlight[mask] = [0, 0, 255]  # BGR format - red highlighting
    
    return highlight, diff_heatmap, f"Changes Detected: {change_percentage:.2f}%", change_percentage

# Generate a PDF report for image comparison with enhanced layout
def generate_pdf(original, corrupt, output, diff_heatmap, report_path, change_percentage, original_filename):
    if original is None or corrupt is None or output is None:
        st.error("Error: One or more images could not be processed for the report.")
        return None

    # Create PDF with more detailed reporting - using A4 size
    pdf = FPDF(orientation='P', unit='mm', format='A4')  # Portrait, millimeters, A4
    pdf.set_auto_page_break(auto=False)  # Disable auto page break to prevent blank pages
    pdf.add_page()
    
    # Define page width and margins for better positioning
    page_width = 210  # A4 width in mm
    page_height = 297  # A4 height in mm
    margin = 10
    content_width = page_width - 2 * margin
    
    # Add header and title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(content_width, 10, txt="3D Image Analysis Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(content_width, 10, txt=f"Comparison Report: {original_filename}", ln=True, align='C')
    pdf.cell(content_width, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    pdf.ln(5)
    
    # Add result summary
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 10, txt="Analysis Result", ln=True)
    pdf.set_font("Arial", size=12)
    
    if change_percentage < 0.01:
        pdf.set_text_color(0, 128, 0)  # Green text
        pdf.cell(content_width, 10, txt="No significant changes detected. The images are identical.", ln=True)
    elif change_percentage < 1:
        pdf.set_text_color(255, 128, 0)  # Orange text
        pdf.cell(content_width, 10, txt=f"Minor differences detected: {change_percentage:.2f}%", ln=True)
    else:
        pdf.set_text_color(255, 0, 0)  # Red text
        pdf.cell(content_width, 10, txt=f"Significant differences detected: {change_percentage:.2f}%", ln=True)

    pdf.ln(5)
    pdf.set_text_color(0, 0, 0)  # Reset text color

    # Save temporary images for PDF - save as PNG for better quality
    temp_dir = "Temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    temp_original = os.path.join(temp_dir, f"temp_original_{int(time.time())}.png")
    temp_corrupt = os.path.join(temp_dir, f"temp_corrupt_{int(time.time())}.png")
    temp_output = os.path.join(temp_dir, f"temp_output_{int(time.time())}.png")
    temp_heatmap = os.path.join(temp_dir, f"temp_heatmap_{int(time.time())}.png")
    
    cv2.imwrite(temp_original, original)
    cv2.imwrite(temp_corrupt, corrupt)
    cv2.imwrite(temp_output, output)
    if diff_heatmap is not None:
        cv2.imwrite(temp_heatmap, diff_heatmap)
    else:
        # Create a dummy heatmap if none provided
        cv2.imwrite(temp_heatmap, output)

    # Calculate image dimensions to fit properly
    img_width = (content_width - 10) / 2  # Two images per row with small gap
    
    # Get aspect ratio from original image to maintain proportions
    aspect_ratio = original.shape[0] / original.shape[1]
    img_height = img_width * aspect_ratio
    
    # Make sure images don't exceed page height
    max_img_height = 50  # Maximum height for images in mm (reduced to prevent overflow)
    if img_height > max_img_height:
        img_height = max_img_height
        img_width = img_height / aspect_ratio
    
    # Add image comparison section heading
    pdf.ln(2)  # Reduced extra space
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 8, txt="Image Comparison", ln=True)
    
    # First row: Labels for Original and Corrupt
    pdf.set_font("Arial", 'B', size=10)
    first_col = margin
    second_col = margin + img_width + 10
    
    # Set Y position for labels
    label_y = pdf.get_y()
    pdf.set_xy(first_col, label_y)
    pdf.cell(img_width, 8, "Approved Image", 0, 0, 'C')
    
    pdf.set_xy(second_col, label_y)
    pdf.cell(img_width, 8, "Corrupt Image", 0, 1, 'C')
    
    # Set Y position for first row of images
    first_row_y = pdf.get_y()
    
    # Place first row images
    pdf.image(temp_original, x=first_col, y=first_row_y, w=img_width, h=img_height)
    pdf.image(temp_corrupt, x=second_col, y=first_row_y, w=img_width, h=img_height)
    
    # Move to position after first row images
    pdf.set_y(first_row_y + img_height + 5)  # Reduced spacing (5mm)
    
    # Second row: Labels for Output and Heatmap
    label_y = pdf.get_y()
    pdf.set_xy(first_col, label_y)
    pdf.cell(img_width, 8, "Differences Highlighted", 0, 0, 'C')
    
    pdf.set_xy(second_col, label_y)
    pdf.cell(img_width, 8, "Difference Heatmap", 0, 1, 'C')
    
    # Set Y position for second row of images
    second_row_y = pdf.get_y()
    
    # Place second row images
    pdf.image(temp_output, x=first_col, y=second_row_y, w=img_width, h=img_height)
    pdf.image(temp_heatmap, x=second_col, y=second_row_y, w=img_width, h=img_height)
    
    # Move to position after second row images
    pdf.set_y(second_row_y + img_height + 10)  # Reduced spacing (10mm)
    
    # Check if we need more space for details
    if pdf.get_y() > page_height - 50:  # If less than 50mm left
        pdf.add_page()  # Add new page for details
    
    # Add analysis details
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 8, txt="Analysis Details", ln=True)
    pdf.set_font("Arial", size=12)
    
    # Create a nice table for analysis details
    detail_y = pdf.get_y()
    line_height = 8
    
    # Row 1: Total pixels
    pdf.set_font("Arial", 'B', size=12)
    pdf.set_xy(margin, detail_y)
    pdf.cell(50, line_height, "Total pixels:", 0, 0)
    
    pdf.set_font("Arial", size=12)
    pdf.set_xy(margin + 50, detail_y)
    pdf.cell(content_width - 50, line_height, f"{original.shape[0] * original.shape[1]:,}", 0, 1)
    
    # Row 2: Changed pixels
    detail_y += line_height
    pdf.set_font("Arial", 'B', size=12)
    pdf.set_xy(margin, detail_y)
    pdf.cell(50, line_height, "Changed pixels:", 0, 0)
    
    pdf.set_font("Arial", size=12)
    pdf.set_xy(margin + 50, detail_y)
    pdf.cell(content_width - 50, line_height, f"{int(change_percentage * original.shape[0] * original.shape[1] / 100):,}", 0, 1)
    
    # Row 3: Change percentage
    detail_y += line_height
    pdf.set_font("Arial", 'B', size=12)
    pdf.set_xy(margin, detail_y)
    pdf.cell(50, line_height, "Change percentage:", 0, 0)
    
    pdf.set_font("Arial", size=12)
    pdf.set_xy(margin + 50, detail_y)
    pdf.cell(content_width - 50, line_height, f"{change_percentage:.4f}%", 0, 1)
    
    # Add footer
    pdf.set_y(page_height - 15)
    pdf.set_font("Arial", 'I', size=8)
    pdf.cell(0, 10, f"3D Image Analysis Tool - Report generated for {original_filename}", 0, 0, 'C')

    # Output the PDF
    try:
        pdf.output(report_path, "F")
        
        # Clean up temporary files after successful PDF generation
        for tmp_file in [temp_original, temp_corrupt, temp_output, temp_heatmap]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        return report_path
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Function to generate consolidated PDF report for batch processing
def generate_consolidated_pdf(summary_data, report_path="Report Folder/consolidated_report.pdf"):
    if not summary_data:
        st.error("No data available for consolidated report")
        return None
    
    # Create PDF
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()
    
    # Define page dimensions
    page_width = 210
    page_height = 297
    margin = 10
    content_width = page_width - 2 * margin
    
    # Add header
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(content_width, 10, txt="3D Image Analysis - Consolidated Report", ln=True, align='C')
    
    pdf.set_font("Arial", size=12)
    pdf.cell(content_width, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
    
    # Add summary section
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 10, txt=f"Summary ({len(summary_data)} images processed)", ln=True)
    
    # Calculate statistics
    changes = [item["change_percentage"] for item in summary_data]
    avg_change = sum(changes) / len(changes) if changes else 0
    max_change = max(changes) if changes else 0
    min_change = min(changes) if changes else 0
    
    # Add statistics
    pdf.set_font("Arial", size=12)
    pdf.cell(content_width, 8, txt=f"Average change: {avg_change:.2f}%", ln=True)
    pdf.cell(content_width, 8, txt=f"Maximum change: {max_change:.2f}%", ln=True)
    pdf.cell(content_width, 8, txt=f"Minimum change: {min_change:.2f}%", ln=True)
    
    # Create table header for results
    pdf.ln(5)
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 10, txt="Detailed Results", ln=True)
    
    # Table header
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', size=12)
    
    # Column widths
    col_widths = [8, 120, 40, 25]  # No., Filename, Change %, Status
    header_texts = ["No.", "Filename", "Change %", "Status"]
    
    # Draw table header
    x_pos = margin
    for i, (width, text) in enumerate(zip(col_widths, header_texts)):
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(width, 10, text, 1, 0, 'C', True)
        x_pos += width
    pdf.ln()
    
    # Table rows
    pdf.set_font("Arial", size=10)
    row_height = 8
    
    for i, item in enumerate(summary_data):
        # Check if we need a new page
        if pdf.get_y() > page_height - 20:
            pdf.add_page()
            
            # Redraw table header on new page
            pdf.set_font("Arial", 'B', size=12)
            x_pos = margin
            for width, text in zip(col_widths, header_texts):
                pdf.set_xy(x_pos, pdf.get_y())
                pdf.cell(width, 10, text, 1, 0, 'C', True)
                x_pos += width
            pdf.ln()
            pdf.set_font("Arial", size=10)
        
        # Row data
        x_pos = margin
        
        # No.
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(col_widths[0], row_height, str(i+1), 1, 0, 'C')
        x_pos += col_widths[0]
        
        # Filename
        filename = item["filename"]
        # Truncate if too long
        if len(filename) > 40:
            display_filename = filename[:37] + "..."
        else:
            display_filename = filename
            
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(col_widths[1], row_height, display_filename, 1, 0, 'L')
        x_pos += col_widths[1]
        
        # Change %
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(col_widths[2], row_height, f"{item['change_percentage']:.2f}%", 1, 0, 'C')
        x_pos += col_widths[2]
        
        # Status
        status = "OK"
        if item["change_percentage"] > 10:
            status = "Major Change"
            pdf.set_text_color(255, 0, 0)
        elif item["change_percentage"] > 1:
            status = "Minor Change"
            pdf.set_text_color(255, 128, 0)
        else:
            pdf.set_text_color(0, 128, 0)
            
        pdf.set_xy(x_pos, pdf.get_y())
        pdf.cell(col_widths[3], row_height, status, 1, 0, 'C')
        pdf.set_text_color(0, 0, 0)  # Reset text color
        
        pdf.ln()
    
    # Add footer
    pdf.set_y(page_height - 15)
    pdf.set_font("Arial", 'I', size=8)
    pdf.cell(0, 10, "3D Image Analysis Tool - Consolidated Report", 0, 0, 'C')
    
    # Output PDF
    try:
        pdf.output(report_path, "F")
        return report_path
    except Exception as e:
        st.error(f"Error generating consolidated PDF: {e}")
        return None

# Batch process images with progress tracking
def batch_process():
    # Get lists of files
    approved_files = os.listdir("Approved Images")
    corrupt_files = os.listdir("Corrupt Images")
    
    # Find intersection of file names in both folders
    common_files = set(approved_files).intersection(set(corrupt_files))
    
    if not common_files:
        st.warning("No matching files found in both Approved and Corrupt Images folders.")
        return
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total_files = len(common_files)
    processed = 0
    
    # Create a summary list for report
    summary = []
    
    for filename in common_files:
        status_text.text(f"Processing {filename}... ({processed+1}/{total_files})")
        
        original_path = os.path.join("Approved Images", filename)
        corrupt_path = os.path.join("Corrupt Images", filename)
        output_path = os.path.join("Output Images", filename)
        report_path = os.path.join("Report Folder", f"{os.path.splitext(filename)[0]}_report.pdf")

        # Load images
        original = cv2.imread(original_path)
        corrupt = cv2.imread(corrupt_path)

        # Check if images are loaded correctly
        if original is None or corrupt is None:
            st.error(f"Error: Could not read {filename}. Skipping...")
            continue

        # Process images
        output, diff_heatmap, change_info, change_percentage = compare_images(original, corrupt)
        
        if output is not None:
            # Save output image
            cv2.imwrite(output_path, output)
            
            # Generate PDF report
            report_file = generate_pdf(original, corrupt, output, diff_heatmap, report_path, change_percentage, filename)
            
            # Add to summary
            summary.append({
                "filename": filename,
                "change_percentage": change_percentage,
                "report": report_file
            })
        
        # Update progress
        processed += 1
        progress_bar.progress(processed / total_files)
    
    # Clear progress display
    progress_bar.empty()
    status_text.empty()
    
    # Display summary
    if summary:
        st.success(f"Batch Processing Complete! {len(summary)} files processed.")
        return summary
    else:
        st.warning("No files were successfully processed.")
        return []

# Clear folders function
def clear_folders(folders=None):
    if folders is None:
        folders = ["Output Images", "Report Folder", "Temp"]
    
    for folder in folders:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                st.error(f"Error clearing {file_path}: {e}")

# Function to apply image enhancement
def enhance_image(image, brightness=0, contrast=1, sharpness=0):
    if image is None:
        return None
        
    # Convert to float for processing
    img_float = image.astype(float)
    
    # Apply brightness
    if brightness != 0:
        img_float += brightness
        
    # Apply contrast
    if contrast != 1:
        img_float = (img_float - 128) * contrast + 128
        
    # Clip values to valid range
    img_float = np.clip(img_float, 0, 255)
    
    # Convert back to uint8
    img = img_float.astype(np.uint8)
    
    # Apply sharpening if requested
    if sharpness > 0:
        kernel = np.array([[-1, -1, -1],
                          [-1, 9 + sharpness, -1],
                          [-1, -1, -1]])
        img = cv2.filter2D(img, -1, kernel)
    
    return img

# Function to apply different visualization modes
def apply_visualization(original, corrupt, mode="standard"):
    if original is None or corrupt is None:
        return None, None, "Error: One or both images could not be loaded!"
    
    if mode == "standard":
        return compare_images(original, corrupt)
    elif mode == "side_by_side":
        # Create side-by-side image
        h, w = original.shape[:2]
        combined = np.zeros((h, w*2, 3), dtype=np.uint8)
        combined[:, :w] = original
        combined[:, w:] = corrupt
        
        # Draw a vertical line between the images
        combined[:, w-1:w+1] = [0, 255, 0]  # Green line
        
        # Calculate difference percentage
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(threshold_diff)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return combined, None, f"Changes Detected: {change_percentage:.2f}%", change_percentage
    elif mode == "difference_only":
        # Show only the difference
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_original, gray_corrupt)
        diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_JET)
        
        # Calculate difference percentage
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(threshold_diff)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return diff_color, None, f"Changes Detected: {change_percentage:.2f}%", change_percentage
    elif mode == "overlay":
        # Create an overlay effect
        alpha = 0.5
        overlay = cv2.addWeighted(original, alpha, corrupt, 1-alpha, 0)
        
        # Calculate difference percentage
        gray_original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        gray_corrupt = cv2.cvtColor(corrupt, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray_original, gray_corrupt)
        _, threshold_diff = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(threshold_diff)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return overlay, None, f"Changes Detected: {change_percentage:.2f}%", change_percentage

# Main application UI
def main():
    st.title("🔍 3D Image Analysis Tool")
    st.sidebar.image("https://via.placeholder.com/150x50.png?text=3D+Analyzer", use_container_width=True)
    
    # Mode selection
    mode = st.sidebar.radio("Select Mode", ("Single Comparison", "Batch Processing", "Settings"))
    
    if mode == "Single Comparison":
        st.sidebar.header("Upload Images")
        
        # File uploaders
        approved_image = st.sidebar.file_uploader("Upload Approved Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        corrupt_image = st.sidebar.file_uploader("Upload Corrupt Image", type=["png", "jpg", "jpeg", "tif", "tiff"])
        
        # Advanced settings
        st.sidebar.header("Analysis Settings")
        threshold = st.sidebar.slider("Difference Threshold", 1, 100, 30, 
                                    help="Lower values detect more subtle differences")
        
        vis_mode = st.sidebar.selectbox("Visualization Mode", 
                                      ["standard", "side_by_side", "difference_only", "overlay"],
                                      help="Choose how to visualize differences")
        
        # Enhancement options
        st.sidebar.header("Image Enhancement")
        with st.sidebar.expander("Enhancement Options"):
            brightness = st.slider("Brightness", -50, 50, 0)
            contrast = st.slider("Contrast", 0.5, 1.5, 1.0, 0.1)
            sharpness = st.slider("Sharpness", 0.0, 2.0, 0.0, 0.1)
            
            apply_to = st.radio("Apply to:", ("None", "Approved Image", "Corrupt Image", "Both"))
        
        # Process the images if both are uploaded
        if approved_image and corrupt_image:
            # Save uploaded files
            approved_path = save_uploaded_file(approved_image, "Approved Images")
            corrupt_path = save_uploaded_file(corrupt_image, "Corrupt Images")
            
            # Load images
            original = load_image(approved_path)
            corrupt = load_image(corrupt_path)
            
            # Apply enhancements if selected
            if apply_to in ["Approved Image", "Both"] and original is not None:
                original = enhance_image(original, brightness, contrast, sharpness)
            
            if apply_to in ["Corrupt Image", "Both"] and corrupt is not None:
                corrupt = enhance_image(corrupt, brightness, contrast, sharpness)
            
            # Process images with selected visualization mode
            output, diff_heatmap, change_info, change_percentage = apply_visualization(original, corrupt, vis_mode)
            
            if output is not None:
                # Display the original images and output
                if vis_mode == "standard":
                    col1, col2, col3 = st.columns(3)
                    col1.image(original, caption="✅ Approved Image", use_container_width=True)
                    col2.image(corrupt, caption="⚠️ Corrupt Image", use_container_width=True)
                    col3.image(output, caption=f"🔴 {change_info}", use_container_width=True)
                    
                    # Display heatmap in an expander
                    with st.expander("View Difference Heatmap"):
                        st.image(diff_heatmap, caption="Difference Heatmap", use_container_width=True)
                else:
                    # For other visualization modes
                    st.image(output, caption=f"{change_info}", use_container_width=True)
                
                # Analysis details
                with st.expander("Analysis Details"):
                    st.write(f"**Image Dimensions:** {original.shape[1]}x{original.shape[0]} pixels")
                    st.write(f"**Total Pixels:** {original.shape[0] * original.shape[1]}")
                    st.write(f"**Changed Pixels:** {int(change_percentage * original.shape[0] * original.shape[1] / 100)}")
                    st.write(f"**Change Percentage:** {change_percentage:.4f}%")
                    
                    if change_percentage < 0.01:
                        st.success("The images are identical or have negligible differences.")
                    elif change_percentage < 1:
                        st.info("The images have minor differences.")
                    else:
                        st.warning("The images have significant differences.")
                
                # Generate report button
                if st.button("📄 Generate Report"):
                    with st.spinner("Generating report..."):
                        output_path = os.path.join("Output Images", approved_image.name)
                        cv2.imwrite(output_path, output)
                        
                        report_path = os.path.join("Report Folder", f"{os.path.splitext(approved_image.name)[0]}_report.pdf")
                        report_file = generate_pdf(original, corrupt, output, diff_heatmap if diff_heatmap is not None else output, 
                                                 report_path, change_percentage, approved_image.name)
                        
                        if report_file:
                            with open(report_file, "rb") as file:
                                st.download_button("📥 Download Report", file, file_name=f"{os.path.splitext(approved_image.name)[0]}_report.pdf", 
                                                mime="application/pdf")
            else:
                st.error("Failed to process images. Please check the uploaded files.")
                
    elif mode == "Batch Processing":
        st.subheader("Batch Image Processing")
        
        # Instructions
        st.info("""
        1. Place your approved images in the 'Approved Images' folder
        2. Place your corrupt images in the 'Corrupt Images' folder 
        3. Files with the same name will be compared
        4. Click 'Start Batch Processing' to begin
        """)
        
        # Advanced settings for batch mode
        st.sidebar.header("Batch Settings")
        threshold = st.sidebar.slider("Difference Threshold", 1, 100, 30, 
                                    help="Lower values detect more subtle differences")
        
        col1, col2 = st.columns(2)
        
        # Start batch processing
        if col1.button("🚀 Start Batch Processing"):
            with st.spinner("Processing images..."):
                summary = batch_process()
                
                if summary:
                    # Display summary table
                    summary_data = {
                        "Filename": [item["filename"] for item in summary],
                        "Change %": [f"{item['change_percentage']:.2f}%" for item in summary]
                    }
                    st.table(summary_data)
        
        # Clear output folders
        if col2.button("🗑️ Clear Output Folders"):
            with st.spinner("Clearing folders..."):
                clear_folders()
                st.success("Output folders cleared successfully.")
        
        # Display available reports
        st.subheader("📂 Available Reports")
        reports = os.listdir("Report Folder")
        
        if reports:
            # Sort reports by creation time
            reports.sort(key=lambda x: os.path.getctime(os.path.join("Report Folder", x)), reverse=True)
            
            # Group by 3 in columns
            for i in range(0, len(reports), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i+j < len(reports):
                        report = reports[i+j]
                        report_path = os.path.join("Report Folder", report)
                        with open(report_path, "rb") as file:
                            cols[j].download_button(f"📥 {report}", file, file_name=report, mime="application/pdf")
        else:
            st.info("No reports available.")
    
    elif mode == "Settings":
        st.subheader("Application Settings")
        
        # Directory management
        st.write("### Directory Management")
        
        # Show directory contents
        with st.expander("View Directory Contents"):
            dir_to_view = st.selectbox("Select Directory", 
                                     ["Approved Images", "Corrupt Images", "Output Images", "Report Folder"])
            
            files = os.listdir(dir_to_view)
            if files:
                st.write(f"**Files in {dir_to_view}:**")
                for file in files:
                    st.text(f"- {file}")
            else:
                st.info(f"No files in {dir_to_view}")
        
        # Clear specific directories
        with st.expander("Clear Directories"):
            dirs_to_clear = st.multiselect("Select directories to clear", 
                                         ["Output Images", "Report Folder", "Approved Images", "Corrupt Images", "Temp"])
            
            if st.button("Clear Selected Directories") and dirs_to_clear:
                with st.spinner("Clearing directories..."):
                    clear_folders(dirs_to_clear)
                    st.success(f"Cleared: {', '.join(dirs_to_clear)}")
        
        # About section
        st.write("### About")
        st.write("""
        **3D Image Analysis Tool**
        
        Version: 1.1.0
        
        This tool helps you analyze and compare 3D model images to detect differences 
        between approved and potentially corrupt versions. 
        
        Features:
        - Single image comparison with multiple visualization modes
        - Batch processing for multiple images
        - Detailed PDF reports
        - Image enhancement options
        - Difference highlighting and heatmaps
        """)

# Run the application
if __name__ == "__main__":
    main()