from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain.docstore.document import Document # type: ignore
from PyPDF2 import PdfReader # type: ignore
import pytesseract # type: ignore
#pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Phanindra BJ\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'
from pdf2image import convert_from_path # type: ignore
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import cv2
import io
import os
import json
import re
import fitz  # PyMuPDF for better PDF handling # type: ignore
import shutil  # For directory removal
import glob    # For file pattern matching

# Set up pytesseract path if needed
# pytesseract.pytesseract.tesseract_cmd = r'path_to_tesseract_executable'

# Function to clean up previous run artifacts
def cleanup_previous_artifacts(index_path, pdf_name):
    """Delete previous FAISS index directory and generated JSON files"""
    # Delete previous FAISS index directory if it exists
    if os.path.exists(index_path):
        print(f"Removing previous FAISS index directory: {index_path}")
        try:
            shutil.rmtree(index_path)
            print(f"Successfully deleted {index_path}")
        except Exception as e:
            print(f"Error deleting directory {index_path}: {e}")
    
    # Find and delete any JSON files matching the PDF name pattern
    json_pattern = f"{os.path.splitext(pdf_name)[0]}*_structured.json"
    json_files = glob.glob(json_pattern)
    if json_files:
        print(f"Removing previous JSON files: {json_files}")
        for json_file in json_files:
            try:
                os.remove(json_file)
                print(f"Successfully deleted {json_file}")
            except Exception as e:
                print(f"Error deleting file {json_file}: {e}")

# Load PDF and extract comprehensive information
def extract_content_from_pdf(pdf_path):
    """Extract text, tables, and image descriptions from PDF"""
    # Text content from traditional extraction
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"
    
    # Use PyMuPDF for enhanced extraction
    doc = fitz.open(pdf_path)
    
    # Store all extracted content
    all_content = []
    
    # Process each page
    for page_num, page in enumerate(doc):
        page_content = {
            "page_number": page_num + 1,
            "text": page.get_text(),
            "tables": [],
            "images": [],
            "charts": []
        }
        
        # Extract tables using text analysis and structure recognition
        tables = extract_tables_from_page(page)
        for idx, table in enumerate(tables):
            table_id = f"page_{page_num+1}_table_{idx+1}"
            table_text = f"TABLE {table_id}:\n{table}\n"
            page_content["tables"].append({
                "id": table_id,
                "content": table
            })
            
        # Extract images and run image analysis
        image_list = extract_images_from_page(page)
        for idx, img_info in enumerate(image_list):
            img_id = f"page_{page_num+1}_image_{idx+1}"
            img_desc = analyze_image(img_info["image"])
            chart_type = detect_chart_type(img_info["image"])
            
            if chart_type:
                # This is a chart/graph
                chart_data = extract_chart_data(img_info["image"], chart_type)
                page_content["charts"].append({
                    "id": img_id,
                    "type": chart_type,
                    "description": img_desc,
                    "data": chart_data
                })
            else:
                # Regular image
                page_content["images"].append({
                    "id": img_id,
                    "description": img_desc
                })
        
        all_content.append(page_content)
    
    # Create a structured representation
    structured_content = {
        "document_name": os.path.basename(pdf_path),
        "total_pages": len(doc),
        "content": all_content
    }
    
    # Save the structured content for reference
    json_output_path = f"{os.path.splitext(pdf_path)[0]}_structured.json"
    with open(json_output_path, "w") as f:
        json.dump(structured_content, f, indent=2)
    print(f"Structured content saved to: {json_output_path}")
    
    # Convert to text chunks for embedding
    text_chunks = []
    
    # Add regular text
    text_chunks.append(raw_text)
    
    # Add tables with context
    for page in all_content:
        for table in page["tables"]:
            table_text = f"TABLE (Page {page['page_number']}): {table['content']}"
            text_chunks.append(table_text)
    
    # Add chart descriptions with context
    for page in all_content:
        for chart in page["charts"]:
            chart_text = f"CHART (Page {page['page_number']}, Type: {chart['type']}): {chart['description']}. Data values: {chart['data']}"
            text_chunks.append(chart_text)
    
    # Add image descriptions with context
    for page in all_content:
        for img in page["images"]:
            img_text = f"IMAGE (Page {page['page_number']}): {img['description']}"
            text_chunks.append(img_text)
    
    return "\n\n".join(text_chunks), structured_content

def extract_tables_from_page(page):
    """Extract tables from a PDF page using text pattern analysis"""
    tables = []
    
    # Get text with its position info
    text_blocks = page.get_text("dict")["blocks"]
    
    # Look for potential table structures
    # This is a simplified approach - tables typically have text in grid-like positions
    
    # First, extract all text lines with their coordinates
    lines = []
    for block in text_blocks:
        if block.get("lines"):
            for line in block["lines"]:
                if line.get("spans"):
                    text = "".join(span["text"] for span in line["spans"])
                    bbox = line["bbox"]  # [x0, y0, x1, y1]
                    lines.append({
                        "text": text,
                        "y": bbox[1],  # Top Y-coordinate
                        "x": bbox[0],  # Left X-coordinate
                        "width": bbox[2] - bbox[0],
                        "height": bbox[3] - bbox[1]
                    })
    
    # Group lines that might be in the same table based on position and structure
    potential_tables = []
    current_table_lines = []
    
    # Sort lines by vertical position
    lines = sorted(lines, key=lambda x: x["y"])
    
    # Simple table detection heuristic - lines close together with similar structure
    for i, line in enumerate(lines):
        # Check if line seems to be part of a table (has multiple spaces or tab-like spacing)
        if re.search(r'\s{2,}|\t', line["text"]) or '|' in line["text"]:
            current_table_lines.append(line["text"])
            
            # If this is the last line or next line is far away, end this table
            if i == len(lines) - 1 or abs(lines[i+1]["y"] - line["y"]) > 2 * line["height"]:
                if len(current_table_lines) >= 2:  # At least 2 rows to form a table
                    potential_tables.append(current_table_lines)
                current_table_lines = []
    
    # Process each potential table
    for table_lines in potential_tables:
        # Convert to a more structured format
        table_text = "\n".join(table_lines)
        tables.append(table_text)
    
    return tables

def extract_images_from_page(page):
    """Extract images from a PDF page"""
    images = []
    
    # Get image list from the page
    img_list = page.get_images(full=True)
    
    for img_index, img_info in enumerate(img_list):
        xref = img_info[0]
        
        # Extract image
        base_image = page.parent.extract_image(xref)
        image_bytes = base_image["image"]
        
        # Convert to OpenCV format for analysis
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        images.append({
            "index": img_index,
            "image": img
        })
    
    return images

def analyze_image(img):
    """Extract text and analyze image content"""
    # Convert to grayscale for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Extract text from image using OCR
    try:
        text = pytesseract.image_to_string(gray)
        if text.strip():
            return f"Image contains text: {text.strip()}"
        else:
            # If no text, provide basic image analysis
            height, width, channels = img.shape
            colors = detect_dominant_colors(img)
            return f"Image size: {width}x{height}. Dominant colors: {colors}."
    except Exception as e:
        return "Image (unable to analyze details)"

def detect_dominant_colors(img, num_colors=3):
    """Detect the dominant colors in an image"""
    # Reshape image to be a list of pixels
    pixels = img.reshape(-1, 3)
    
    # Convert to float for k-means
    pixels = np.float32(pixels)
    
    # Define criteria and apply kmeans
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    # Convert back to uint8
    centers = np.uint8(centers)
    
    # Count occurrences of each label
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sort by occurrence (most frequent first)
    sorted_indices = np.argsort(-counts)
    sorted_centers = centers[sorted_indices]
    
    # Format as RGB strings
    color_names = []
    for center in sorted_centers:
        b, g, r = center
        color_names.append(f"RGB({r},{g},{b})")
    
    return ", ".join(color_names[:num_colors])

def detect_chart_type(img):
    """Detect if image is a chart and what type"""
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Check for chart-specific features
    
    # Extract text in image using OCR for clues
    text = pytesseract.image_to_string(gray).lower()
    
    # Check for chart-related keywords in OCR text
    chart_keywords = {
        "bar chart": ["bar chart", "bar graph"],
        "line chart": ["line chart", "line graph", "trend"],
        "pie chart": ["pie chart", "pie graph", "percentage", "portion"],
        "scatter plot": ["scatter", "correlation"],
        "heat map": ["heat map", "heatmap", "temperature map"],
        "histogram": ["histogram", "distribution", "frequency"],
        "table": ["table", "grid", "column", "row"]
    }
    
    for chart_type, keywords in chart_keywords.items():
        if any(keyword in text for keyword in keywords):
            return chart_type
    
    # Visual feature-based detection (simplified)
    # Count horizontal and vertical lines
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    
    if lines is not None:
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > abs(y2 - y1):
                horizontal_lines += 1
            else:
                vertical_lines += 1
        
        # Simple heuristics
        if horizontal_lines > 5 and vertical_lines > 5:
            return "table"
        elif horizontal_lines > vertical_lines * 2:
            return "bar chart"
        elif vertical_lines > horizontal_lines * 2:
            return "column chart"
    
    # Check for circles (pie chart)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=20, maxRadius=200)
    if circles is not None and len(circles[0]) > 0:
        return "pie chart"
    
    # Default if we can't determine
    return None

def extract_chart_data(img, chart_type):
    """Extract data from charts based on type"""
    # This is a simplified implementation
    # Real chart data extraction requires more sophisticated computer vision
    
    # Run OCR to get all text in the image
    text = pytesseract.image_to_string(img)
    
    # Look for numbers in the text
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    
    if chart_type == "pie chart":
        # Look for percentage symbols
        percentages = re.findall(r'\b\d+(?:\.\d+)?%', text)
        return f"Pie chart with values: {', '.join(percentages if percentages else numbers[:5])}"
    
    elif chart_type in ["bar chart", "column chart", "line chart"]:
        # Try to find axis labels
        lines = text.split('\n')
        axis_labels = []
        for line in lines:
            if len(line.strip()) > 0 and not re.match(r'^\d+(?:\.\d+)?$', line.strip()):
                axis_labels.append(line.strip())
        
        # Combine data points and labels
        if len(axis_labels) > 0:
            return f"Chart with labels [{', '.join(axis_labels[:3])}...] and values [{', '.join(numbers[:5])}...]"
        else:
            return f"Chart with values: {', '.join(numbers[:5])}"
    
    elif chart_type == "heat map":
        return f"Heat map with intensity values ranging from {min(float(n) for n in numbers) if numbers else 'unknown'} to {max(float(n) for n in numbers) if numbers else 'unknown'}"
    
    elif chart_type == "table":
        # For tables, return a sample of the detected text
        lines = text.split('\n')
        clean_lines = [line for line in lines if line.strip()]
        if clean_lines:
            sample = clean_lines[:3]
            return f"Table with {len(clean_lines)} rows. Sample: {'; '.join(sample)}"
    
    return f"Data values: {', '.join(numbers[:5]) if numbers else 'Unable to extract specific values'}"

# Chunk text with metadata
def chunk_text_with_metadata(text, structured_content):
    """Split text into chunks while preserving metadata"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    raw_chunks = splitter.split_text(text)
    
    # Create Document objects with metadata
    documents = []
    
    for i, chunk in enumerate(raw_chunks):
        # Determine metadata based on content patterns
        metadata = {"chunk_id": i}
        
        if "TABLE" in chunk:
            metadata["content_type"] = "table"
            # Extract page number if available
            page_match = re.search(r"TABLE \(Page (\d+)", chunk)
            if page_match:
                metadata["page"] = int(page_match.group(1))
        
        elif "CHART" in chunk:
            metadata["content_type"] = "chart"
            # Extract chart type if available
            type_match = re.search(r"Type: ([^)]+)", chunk)
            if type_match:
                metadata["chart_type"] = type_match.group(1)
            
            # Extract page number
            page_match = re.search(r"CHART \(Page (\d+)", chunk)
            if page_match:
                metadata["page"] = int(page_match.group(1))
        
        elif "IMAGE" in chunk:
            metadata["content_type"] = "image"
            # Extract page number
            page_match = re.search(r"IMAGE \(Page (\d+)", chunk)
            if page_match:
                metadata["page"] = int(page_match.group(1))
        
        else:
            metadata["content_type"] = "text"
        
        documents.append(Document(page_content=chunk, metadata=metadata))
    
    return documents

# Generate FAISS index with metadata
def create_faiss_index(documents, index_path):
    """Create FAISS index from document objects with metadata"""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    vectorstore.save_local(index_path)
    print(f"FAISS index saved to: {index_path}")

if __name__ == "__main__":
    pdf_path = "BEAT-article_thebeatmar2025.pdf"
    index_path = "beat_article_1"

    print("Starting process with cleanup...")
    # First clean up previous artifacts
    cleanup_previous_artifacts(index_path, pdf_path)
    
    print("Extracting content from PDF...")
    raw_text, structured_content = extract_content_from_pdf(pdf_path)
    
    print("Creating chunks with metadata...")
    document_chunks = chunk_text_with_metadata(raw_text, structured_content)
    
    print(f"Created {len(document_chunks)} chunks")
    print("Generating FAISS index...")
    create_faiss_index(document_chunks, index_path)
    
    print("Process completed successfully!")
