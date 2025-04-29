# Lab Report Processing API

This API processes lab report images to extract lab test names, values, and reference ranges. It is built using FastAPI and doesn't utilize any LLMs as per the requirements.

## Features

- Extracts lab test names from medical lab reports
- Identifies test values and units
- Determines reference ranges for each test
- Flags tests that are out of range
- Provides a simple API endpoint for integration

## Installation

### Prerequisites

- Python 3.8+
- Tesseract OCR

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd lab-report-api
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Tesseract OCR:
   - **Linux**: `apt-get install tesseract-ocr`
   - **Windows**: Download from [Tesseract GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
   - **macOS**: `brew install tesseract`

## Running the API

### Local Development

```bash
uvicorn main:app --reload
```

### Production Deployment

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

## Docker Deployment

1. Build the Docker image:
```bash
docker build -t lab-report-api .
```

2. Run the container:
```bash
docker run -p 8000:8000 lab-report-api
```

## API Endpoints

### GET /
Returns a simple message confirming the API is running.

### POST /get-lab-tests
Processes a
