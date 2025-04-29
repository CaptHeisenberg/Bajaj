import os
import cv2
import numpy as np
import pytesseract
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import re
from PIL import Image
import io
import uvicorn

app = FastAPI(title="Lab Report Processing API")

class LabTest(BaseModel):
    test_name: str
    test_value: str
    bio_reference_range: Optional[str] = None
    lab_test_out_of_range: Optional[bool] = None
    unit: Optional[str] = None

class LabReportResponse(BaseModel):
    is_success: bool
    lab_tests: List[LabTest] = []
    error_message: Optional[str] = None

def preprocess_image(image_bytes):
    """Preprocess the image for better OCR results."""
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    
    # Apply noise reduction
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    
    return denoised

def extract_text_from_image(image_bytes):
    """Extract text from image using OCR."""
    try:
        # Preprocess the image
        processed_img = preprocess_image(image_bytes)
        
        # Use OCR to extract text
        text = pytesseract.image_to_string(processed_img)
        return text
    except Exception as e:
        raise Exception(f"Error in OCR text extraction: {str(e)}")

def parse_reference_range(range_text):
    """Parse reference range from text."""
    if not range_text:
        return None
    
    # Common range patterns
    range_patterns = [
        r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)',  # 10-20, 1.5-2.5
        r'<\s*(\d+\.?\d*)',                # <10
        r'>\s*(\d+\.?\d*)',                # >10
        r'≤\s*(\d+\.?\d*)',                # ≤10
        r'≥\s*(\d+\.?\d*)'                 # ≥10
    ]
    
    for pattern in range_patterns:
        match = re.search(pattern, range_text)
        if match:
            return range_text.strip()
    
    return range_text.strip() if range_text.strip() else None

def extract_unit(value_text):
    """Extract unit from value text."""
    # Common unit patterns
    unit_patterns = [
        r'\d+\.?\d*\s*([a-zA-Z/%]+)',
        r'\d+\.?\d*\s*([µμnmcdl]+)'
    ]
    
    for pattern in unit_patterns:
        match = re.search(pattern, value_text)
        if match:
            return match.group(1).strip()
    
    return None

def is_out_of_range(value, reference_range):
    """Check if test value is outside reference range."""
    if not reference_range:
        return None
    
    try:
        # Extract numeric value
        value_match = re.search(r'(\d+\.?\d*)', value)
        if not value_match:
            return None
        
        numeric_value = float(value_match.group(1))
        
        # Check different range patterns
        range_match = re.search(r'(\d+\.?\d*)\s*-\s*(\d+\.?\d*)', reference_range)
        if range_match:
            lower = float(range_match.group(1))
            upper = float(range_match.group(2))
            return numeric_value < lower or numeric_value > upper
        
        less_than_match = re.search(r'<\s*(\d+\.?\d*)', reference_range)
        if less_than_match:
            upper = float(less_than_match.group(1))
            return numeric_value >= upper
        
        greater_than_match = re.search(r'>\s*(\d+\.?\d*)', reference_range)
        if greater_than_match:
            lower = float(greater_than_match.group(1))
            return numeric_value <= lower
        
        less_equal_match = re.search(r'≤\s*(\d+\.?\d*)', reference_range)
        if less_equal_match:
            upper = float(less_equal_match.group(1))
            return numeric_value > upper
        
        greater_equal_match = re.search(r'≥\s*(\d+\.?\d*)', reference_range)
        if greater_equal_match:
            lower = float(greater_equal_match.group(1))
            return numeric_value < lower
        
    except Exception:
        pass
    
    return None

def extract_lab_tests(text):
    """Extract lab tests from OCR text."""
    lab_tests = []
    
    # Split text into lines
    lines = text.strip().split('\n')
    
    # Define patterns for test data
    test_patterns = [
        # Pattern for test name, value, and reference range in same line
        r'([A-Za-z\s]+)[\s:]+(\d+\.?\d*\s*[a-zA-Z/%µμ]*)[\s:]+(?:Reference:)?\s*([<>≤≥\d\.\s-]+)',
        
        # Pattern for test name followed by value
        r'([A-Za-z\s]+)[\s:]+(\d+\.?\d*\s*[a-zA-Z/%µμ]*)'
    ]
    
    # Process each line
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Try to match test patterns
        matched = False
        for pattern in test_patterns:
            match = re.search(pattern, line)
            if match:
                test_name = match.group(1).strip()
                test_value = match.group(2).strip()
                
                # Extract reference range if available in the match
                reference_range = None
                if len(match.groups()) > 2:
                    reference_range = match.group(3).strip()
                
                # If no reference range in current line, check next line
                if not reference_range and i + 1 < len(lines):
                    next_line = lines[i + 1].strip()
                    ref_match = re.search(r'(?:Reference:)?\s*([<>≤≥\d\.\s-]+)', next_line)
                    if ref_match:
                        reference_range = ref_match.group(1).strip()
                        i += 1  # Skip next line as we've used it
                
                # Parse reference range
                parsed_range = parse_reference_range(reference_range)
                
                # Extract unit
                unit = extract_unit(test_value)
                
                # Check if value is out of range
                out_of_range = is_out_of_range(test_value, parsed_range)
                
                # Create lab test object
                lab_test = LabTest(
                    test_name=test_name,
                    test_value=test_value,
                    bio_reference_range=parsed_range,
                    lab_test_out_of_range=out_of_range,
                    unit=unit
                )
                
                lab_tests.append(lab_test)
                matched = True
                break
        
        if not matched:
            # Check if this line might be a test name and the next line contains value
            if i + 1 < len(lines):
                next_line = lines[i + 1].strip()
                value_match = re.search(r'^(\d+\.?\d*\s*[a-zA-Z/%µμ]*)', next_line)
                if value_match and re.match(r'^[A-Za-z\s]+$', line):
                    test_name = line
                    test_value = value_match.group(1).strip()
                    
                    # Check for reference range in next lines
                    reference_range = None
                    if i + 2 < len(lines):
                        ref_line = lines[i + 2].strip()
                        ref_match = re.search(r'(?:Reference:)?\s*([<>≤≥\d\.\s-]+)', ref_line)
                        if ref_match:
                            reference_range = ref_match.group(1).strip()
                            i += 2  # Skip next two lines
                        else:
                            i += 1  # Skip just the value line
                    else:
                        i += 1  # Skip just the value line
                    
                    # Parse reference range
                    parsed_range = parse_reference_range(reference_range)
                    
                    # Extract unit
                    unit = extract_unit(test_value)
                    
                    # Check if value is out of range
                    out_of_range = is_out_of_range(test_value, parsed_range)
                    
                    # Create lab test object
                    lab_test = LabTest(
                        test_name=test_name,
                        test_value=test_value,
                        bio_reference_range=parsed_range,
                        lab_test_out_of_range=out_of_range,
                        unit=unit
                    )
                    
                    lab_tests.append(lab_test)
                    matched = True
        
        i += 1
    
    return lab_tests

@app.post("/get-lab-tests", response_model=LabReportResponse)
async def get_lab_tests(file: UploadFile = File(...)):
    """
    Process a lab report image and extract lab test information.
    
    Parameters:
    - file: The lab report image file
    
    Returns:
    - JSON with lab test data including test names, values, and reference ranges
    """
    try:
      
        contents = await file.read()
      
        extracted_text = extract_text_from_image(contents)
        
      
        lab_tests = extract_lab_tests(extracted_text)
        
        # Return response
        return LabReportResponse(
            is_success=True,
            lab_tests=lab_tests
        )
    
    except Exception as e:
        return LabReportResponse(
            is_success=False,
            error_message=str(e)
        )

@app.get("/")
def read_root():
    return {"message": "Lab Report Processing API is running. Use /get-lab-tests endpoint to process images."}

if __name__ == "__main__":
    # Make sure Tesseract OCR is installed and configured
    # For Linux: apt-get install tesseract-ocr
    # For Windows: Download and install from https://github.com/UB-Mannheim/tesseract/wiki
    uvicorn.run(app, host="0.0.0.0", port=8000)
