# SmartDoc AI 📄🤖

SmartDoc AI is an AI-powered document understanding application that extracts text from images, PDFs, and camera captures using OCR and helps users understand and interact with documents using Google Gemini AI.

## Features

- English and Hindi OCR
- Multiple image upload
- PDF document processing
- Camera OCR
- Editable extracted text
- Document statistics
- AI-generated document summary
- Simple explanation
- Key points and keywords extraction
- Document type identification
- Action item detection
- Downloadable AI analysis report
- Multi-turn document chat
- File size validation
- Reset controls
- User-friendly API and network error handling

## Tech Stack

- Python
- Streamlit
- Tesseract OCR
- OpenCV
- Google Gemini API
- NumPy
- Pillow
- pdf2image

## How It Works

1. Upload images, a PDF, or capture a document using the camera.
2. SmartDoc AI extracts text using OCR.
3. Review and correct the extracted text.
4. Generate structured AI document analysis.
5. Chat with the document and ask follow-up questions.
6. Download the extracted text or complete AI report.

## Project Setup

Clone the repository:

git clone YOUR_REPOSITORY_URL

Move into the project folder:

cd SmartDoc-AI

Install dependencies:

pip install -r requirements.txt

Create this file:

.streamlit/secrets.toml

Add your Gemini API key:

GEMINI_API_KEY = "YOUR_API_KEY"

Run the application:

streamlit run aiocr.py

## Security

API keys and secrets are not included in this repository. The `.streamlit/secrets.toml` file is excluded using `.gitignore`.

## Future Improvements

- Direct text extraction from digital PDFs
- OCR fallback for scanned PDFs
- Improved support for large documents
- Cloud deployment support

## Author

Vasu Jain

BCA Student | AI/ML & Software Development Enthusiast