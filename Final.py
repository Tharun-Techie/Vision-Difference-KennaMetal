import cv2
from datetime import datetime
from fpdf import FPDF
import numpy as np
import os
from PIL import Image
import streamlit as st
import shutil
import requests
import uuid



# Set page configuration
st.set_page_config(
    page_title="3d CAD Image Vision Difference Analysis ",
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
    pdf.set_auto_page_break(auto=False)  # Enable auto page break with margin
    pdf.add_page()
    
    # Define page width and margins for better positioning
    page_width = 210  # A4 width in mm
    page_height = 297  # A4 height in mm
    margin = 10
    content_width = page_width - 2 * margin
    
    # Add logo to the top right
    temp_logo_path = os.path.join("Temp", "logo.jpg")
    
    # Download and save logo if it doesn't exist
    if not os.path.exists(temp_logo_path):
        try:
            logo_url = "https://mma.prnewswire.com/media/2129550/4843326/Kennametal_Logo.jpg?p=distribution"
            response = requests.get(logo_url)
            if response.status_code == 200:
                with open(temp_logo_path, "wb") as f:
                    f.write(response.content)
        except Exception as e:
            st.warning(f"Could not download logo: {e}")
    
    # Add logo if available
    if os.path.exists(temp_logo_path):
        # Position logo at top right
        pdf.image(temp_logo_path, x=page_width-60, y=10, w=40)
    
    # Add header and title
    pdf.set_font("Arial", 'B', size=16)
    pdf.cell(content_width-50, 10, txt="3D Image Analysis Report", ln=True, align='L')  # Left align to make room for logo
    
    pdf.set_font("Arial", size=12)
    pdf.cell(content_width, 10, txt=f"Comparison Report: {original_filename}", ln=True, align='L')
    pdf.cell(content_width, 10, txt=f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='L')
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
    # Ensure temp directory exists
    temp_dir = "Temp"
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_original = os.path.join(temp_dir, "temp_original.png")
    temp_corrupt = os.path.join(temp_dir, "temp_corrupt.png")
    temp_output = os.path.join(temp_dir, "temp_output.png")
    temp_heatmap = os.path.join(temp_dir, "temp_heatmap.png")
    
    # Convert BGR (OpenCV) to RGB before saving to maintain correct colors
    original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    corrupt_rgb = cv2.cvtColor(corrupt, cv2.COLOR_BGR2RGB)
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    
    # If heatmap is provided, convert it as well
    if diff_heatmap is not None:
        diff_heatmap_rgb = cv2.cvtColor(diff_heatmap, cv2.COLOR_BGR2RGB)
    else:
        diff_heatmap_rgb = output_rgb  # Use output as fallback if no heatmap
    
    # Save the RGB images for PDF
    cv2.imwrite(temp_original, original_rgb)
    cv2.imwrite(temp_corrupt, corrupt_rgb)
    cv2.imwrite(temp_output, output_rgb)
    cv2.imwrite(temp_heatmap, diff_heatmap_rgb)
    
    # Add these checks:
    if not os.path.exists(temp_original) or not os.path.exists(temp_corrupt) or not os.path.exists(temp_output) or not os.path.exists(temp_heatmap):
        st.error("Failed to save temporary images for PDF report")
        return None

    # Calculate image dimensions to fit properly - make images smaller to fit on one page
    img_width = (content_width - 10) / 2  # Two images per row with small gap
    
    # Get aspect ratio from original image to maintain proportions
    aspect_ratio = original.shape[0] / original.shape[1]
    img_height = img_width * aspect_ratio
    
    # Restrict image height to ensure everything fits on one page
    max_img_height = 60  # Reduced maximum height for images in mm
    if img_height > max_img_height:
        img_height = max_img_height
        img_width = img_height / aspect_ratio
    
    # Add image comparison section heading
    pdf.set_font("Arial", 'B', size=14)
    pdf.cell(content_width, 8, txt="Image Comparison", ln=True)
    
    # First row: Labels for Original and Corrupt
    pdf.set_font("Arial", 'B', size=10)
    first_col = margin
    second_col = margin + img_width + 10
    
    # Set Y position for labels
    label_y = pdf.get_y()
    pdf.set_xy(first_col, label_y)
    pdf.cell(img_width, 6, "Approved Image", 0, 0, 'C')
    
    pdf.set_xy(second_col, label_y)
    pdf.cell(img_width, 6, "Corrupt Image", 0, 1, 'C')
    
    # Set Y position for first row of images
    first_row_y = pdf.get_y()
    
    # Place first row images
    pdf.image(temp_original, x=first_col, y=first_row_y, w=img_width, h=img_height)
    pdf.image(temp_corrupt, x=second_col, y=first_row_y, w=img_width, h=img_height)
    
    # Move to position after first row images
    pdf.set_y(first_row_y + img_height + 3)  # Reduced spacing further
    
    # Second row: Labels for Output and Heatmap
    label_y = pdf.get_y()
    pdf.set_xy(first_col, label_y)
    pdf.cell(img_width, 6, "Differences Highlighted", 0, 0, 'C')
    
    pdf.set_xy(second_col, label_y)
    pdf.cell(img_width, 6, "Difference Heatmap", 0, 1, 'C')
    
    # Set Y position for second row of images
    second_row_y = pdf.get_y()
    
    # Place second row images
    pdf.image(temp_output, x=first_col, y=second_row_y, w=img_width, h=img_height)
    pdf.image(temp_heatmap, x=second_col, y=second_row_y, w=img_width, h=img_height)
    
    # Move to position after second row images
    pdf.set_y(second_row_y + img_height + 3)  # Reduced spacing further
    
    # Add analysis details in more condensed format
    pdf.set_font("Arial", 'B', size=12)
    pdf.cell(content_width, 6, txt="Analysis Details", ln=True)
    
    # Create a condensed table for analysis details
    pdf.set_font("Arial", size=10)
    detail_y = pdf.get_y()
    line_height = 5  # Further reduced line height
    
    # Squeeze details into two columns for space efficiency
    col_width = content_width / 2
    
    # Row 1: Total pixels and Changed pixels
    pdf.set_xy(margin, detail_y)
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(40, line_height, "Total pixels:", 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.cell(col_width - 40, line_height, f"{original.shape[0] * original.shape[1]:,}", 0, 0)
    
    pdf.set_xy(margin + col_width, detail_y)
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(40, line_height, "Changed pixels:", 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.cell(col_width - 40, line_height, f"{int(change_percentage * original.shape[0] * original.shape[1] / 100):,}", 0, 1)
    
    # Row 2: Change percentage and dimensions
    detail_y += line_height
    pdf.set_xy(margin, detail_y)
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(40, line_height, "Change percentage:", 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.cell(col_width - 40, line_height, f"{change_percentage:.4f}%", 0, 0)
    
    pdf.set_xy(margin + col_width, detail_y)
    pdf.set_font("Arial", 'B', size=10)
    pdf.cell(40, line_height, "Dimensions:", 0, 0)
    pdf.set_font("Arial", size=10)
    pdf.cell(col_width - 40, line_height, f"{original.shape[1]} x {original.shape[0]} px", 0, 1)
    
    # Add footer
    footer_y = min(280, pdf.get_y() + 10)  # Position footer near bottom of page but not off-page
    pdf.set_y(footer_y)
    pdf.set_font("Arial", 'I', size=8)
    pdf.cell(0, 10, f"3D Image Analysis Tool - Report generated for {original_filename}", 0, 0, 'C')

    # Output the PDF
    try:
        # First ensure the directory exists
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        
        pdf.output(report_path, "F")
        
        # Clean up temporary files after successful PDF generation
        for tmp_file in [temp_original, temp_corrupt, temp_output, temp_heatmap]:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
                
        return report_path
    except Exception as e:
        st.error(f"Error generating PDF: {e}")
        return None

# Batch process images with progress tracking

def batch_process(approved_dir="Approved Images", corrupt_dir="Corrupt Images", threshold_value=30):
    # Get lists of files
    approved_files = os.listdir(approved_dir)
    corrupt_files = os.listdir(corrupt_dir)
    
    # Find intersection of file names in both folders
    common_files = set(approved_files).intersection(set(corrupt_files))
    
    if not common_files:
        st.warning(f"No matching files found in both {approved_dir} and {corrupt_dir} folders.")
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
        
        original_path = os.path.join(approved_dir, filename)
        corrupt_path = os.path.join(corrupt_dir, filename)
        output_path = os.path.join("Output Images", filename)
        report_path = os.path.join("Report Folder", f"{os.path.splitext(filename)[0]}_report.pdf")

        # Load images
        original = load_image(original_path)
        corrupt = load_image(corrupt_path)

        # Check if images are loaded correctly
        if original is None or corrupt is None:
            st.error(f"Error: Could not read {filename}. Skipping...")
            continue

        # Process images - use standard visualization mode for batch processing
        output, diff_heatmap, change_info, change_percentage = compare_images(original, corrupt, threshold_value)
        
        if output is not None:
            # Save output image
            cv2.imwrite(output_path, output)
            
            # Generate PDF report using the same function as Single Comparison mode
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
def apply_visualization(original, corrupt, mode="standard", threshold_value=30):
    if original is None or corrupt is None:
        return None, None, "Error: One or both images could not be loaded!"
    
    if mode == "standard":
        return compare_images(original, corrupt, threshold_value)
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
        _, threshold_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
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
        _, threshold_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
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
        _, threshold_diff = cv2.threshold(diff, threshold_value, 255, cv2.THRESH_BINARY)
        total_pixels = diff.size
        changed_pixels = np.count_nonzero(threshold_diff)
        change_percentage = (changed_pixels / total_pixels) * 100
        
        return overlay, None, f"Changes Detected: {change_percentage:.2f}%", change_percentage

# Function to select directory
def select_directory():
    # Get current working directory
    cwd = os.getcwd()
    
    # List subdirectories in the current directory
    subdirs = [d for d in os.listdir(cwd) if os.path.isdir(os.path.join(cwd, d))]
    
    # Add option to select a custom directory
    subdirs.append("Custom Directory...")
    unique_key = str(uuid.uuid4())
    selected_dir = st.selectbox("Select directory", subdirs, key=unique_key)
    # Let user select directory
    # selected_dir = st.selectbox("Select directory", subdirs)
    #selected_dir = st.selectbox("Select directory", subdirs, key="select_directory_1")

    
    if selected_dir == "Custom Directory...":
        # User can input a custom path
        custom_path = st.text_input("Enter custom directory path")
        if custom_path and os.path.isdir(custom_path):
            return custom_path
        elif custom_path:
            st.error("Invalid directory path")
            return None
        else:
            return None
    else:
        return os.path.join(cwd, selected_dir)

# Download and save logo for use in the app
def download_logo():
    logo_url = "https://mma.prnewswire.com/media/2129550/4843326/Kennametal_Logo.jpg?p=distribution"
    logo_path = os.path.join("Temp", "logo.jpg")
    
    # Create Temp directory if it doesn't exist
    os.makedirs("Temp", exist_ok=True)
    
    # Only download if the file doesn't already exist
    if not os.path.exists(logo_path):
        try:
            response = requests.get(logo_url)
            if response.status_code == 200:
                with open(logo_path, "wb") as f:
                    f.write(response.content)
                return logo_path
        except Exception as e:
            st.warning(f"Could not download logo: {e}")
            return None
    else:
        return logo_path
    
def main():
    st.title("🔍 3D CAD Analysis Difference Detector")
    
    # Download and display logo in sidebar
    logo_path = download_logo()
    if logo_path:
        logo_img = Image.open(logo_path)
        # Resize logo to fit nicely in sidebar
        logo_width = 150
        ratio = logo_width / float(logo_img.size[0])
        logo_height = int(float(logo_img.size[1]) * ratio)
        resized_logo = logo_img.resize((logo_width, logo_height))
        st.sidebar.image(resized_logo, use_container_width=False)
    else:
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
            output, diff_heatmap, change_info, change_percentage = apply_visualization(original, corrupt, vis_mode, threshold)
            
            if output is not None:
                # Display the original images and output
                if vis_mode == "standard":
                    col1, col2, col3 = st.columns(3)
                    col1.image(original, caption="✅ Approved Image", use_container_width=True)
                    col2.image(corrupt, caption="⚠️ Corrupt Image", use_container_width=True)
                    col3.image(output, caption=f"🔴 {change_info}", use_container_width=True)
                    
                    # Display heatmap in an expander with same size as other images
                    with st.expander("View Difference Heatmap"):
                        # Create a container with same column layout to maintain consistent image size
                        heatmap_col1, heatmap_col2, heatmap_col3 = st.columns(3)
                        # Use the middle column to center the heatmap
                        heatmap_col2.image(diff_heatmap, caption="Difference Heatmap", use_container_width=True)
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
                        # Use the same images that are displayed on screen for the report
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
        1. Select your approved and corrupt image folders
        2. Files with the same name will be compared
        3. Click 'Start Batch Processing' to begin
        """)
        
        # Directory selection for both approved and corrupt images
        st.subheader("Select Directories")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Approved Images Directory**")
            approved_dir_default = "Approved Images"
            use_default_approved = st.checkbox("Use default directory for approved images", value=True)
            
            if use_default_approved:
                approved_dir = approved_dir_default
                st.info(f"Using default directory: {approved_dir}")
            else:
                approved_dir = select_directory()
                if approved_dir:
                    st.success(f"Selected directory: {approved_dir}")
                else:
                    st.warning("Please select a valid directory")
        
        with col2:
            st.write("**Corrupt Images Directory**")
            corrupt_dir_default = "Corrupt Images"
            use_default_corrupt = st.checkbox("Use default directory for corrupt images", value=True)
            
            if use_default_corrupt:
                corrupt_dir = corrupt_dir_default
                st.info(f"Using default directory: {corrupt_dir}")
            else:
                corrupt_dir = select_directory()
                if corrupt_dir:
                    st.success(f"Selected directory: {corrupt_dir}")
                else:
                    st.warning("Please select a valid directory")
        
        # Advanced settings for batch mode
        st.sidebar.header("Batch Settings")
        threshold = st.sidebar.slider("Difference Threshold", 1, 100, 30, 
                                    help="Lower values detect more subtle differences")
        
        batch_col1, batch_col2 = st.columns(2)
        
        # Start batch processing
        if batch_col1.button("🚀 Start Batch Processing"):
            # Validate directories
            if not approved_dir or not corrupt_dir:
                st.error("Please select valid directories for both approved and corrupt images")
            elif not os.path.isdir(approved_dir) or not os.path.isdir(corrupt_dir):
                st.error("One or both of the selected directories do not exist")
            else:
                with st.spinner("Processing images..."):
                    summary = batch_process(approved_dir, corrupt_dir, threshold)
                    
                    if summary:
                        # Display summary table
                        summary_data = {
                            "Filename": [item["filename"] for item in summary],
                            "Change %": [f"{item['change_percentage']:.2f}%" for item in summary]
                        }
                        st.table(summary_data)
        
        # Clear output folders
        if batch_col2.button("🗑️ Clear Output Folders"):
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
        
        Version: 1.2.0
        
        This tool helps you analyze and compare 3D model images to detect differences 
        between approved and potentially corrupt versions. 
        
        Features:
        - Single image comparison with multiple visualization modes
        - Batch processing for multiple images
        - Custom directory selection for batch processing
        - Detailed PDF reports with company logo
        - Image enhancement options
        - Difference highlighting and heatmaps
        """)

# Run the application
if __name__ == "__main__":
    main()