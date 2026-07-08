import streamlit as st
import pytesseract
import cv2
import numpy as np
import json
import platform

from PIL import Image
from pdf2image import convert_from_bytes
from google import genai


# ============================================================
# APP CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="SmartDoc AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

try:
    client = genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"]
    )
except Exception:
    client = None


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "ai_result": None,
    "analyzed_text": "",
    "chat_history": [],
    "chat_document_id": None,
    "uploader_version": 0,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# CONSTANTS
# ============================================================

MAX_FILE_SIZE_MB = 10


# ============================================================
# HELPERS
# ============================================================

def is_file_size_valid(uploaded_file):
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024
    return uploaded_file.size <= max_size_bytes


def reset_app():

    keys_to_clear = [
        "ai_result",
        "analyzed_text",
        "chat_history",
        "chat_document_id",
        "combined_image_text",
        "combined_pdf_text",
        "camera_text",
    ]

    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

    st.session_state.uploader_version += 1


def show_ai_error(error):

    error_text = str(error).lower()

    if (
        "429" in error_text
        or "quota" in error_text
        or "resource_exhausted" in error_text
    ):
        st.error(
            "Gemini API quota exceeded. "
            "Please wait and try again later."
        )

    elif (
        "connection" in error_text
        or "network" in error_text
        or "timeout" in error_text
        or "timed out" in error_text
    ):
        st.error(
            "Network connection problem. "
            "Check your internet connection and try again."
        )

    elif (
        "api key" in error_text
        or "api_key" in error_text
        or "401" in error_text
        or "unauthenticated" in error_text
    ):
        st.error(
            "Gemini API authentication failed. "
            "Check your API key."
        )

    elif (
        "503" in error_text
        or "service unavailable" in error_text
    ):
        st.error(
            "Gemini is temporarily unavailable. "
            "Please try again shortly."
        )

    else:
        st.error(
            f"AI request failed: {error}"
        )


# ============================================================
# OCR
# ============================================================

def preprocess(img):

    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    gray = cv2.convertScaleAbs(
        gray,
        alpha=1.6,
        beta=20
    )

    blur = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )


def run_ocr(image):

    img = np.array(
        image.convert("RGB")
    )

    processed = preprocess(img)

    text = pytesseract.image_to_string(
        processed,
        lang="eng+hin",
        config=(
            "--oem 3 "
            "--psm 6 "
            "-c preserve_interword_spaces=1"
        )
    )

    return text.strip()


# ============================================================
# STATISTICS
# ============================================================

def get_document_statistics(text):

    words = len(text.split())
    characters = len(text)
    lines = len(text.splitlines())

    reading_time = (
        0 if words == 0
        else max(1, round(words / 200))
    )

    return {
        "words": words,
        "characters": characters,
        "lines": lines,
        "reading_time": reading_time
    }


def display_statistics(text):

    stats = get_document_statistics(text)

    st.markdown("### 📊 Document Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Words",
        stats["words"]
    )

    col2.metric(
        "Characters",
        stats["characters"]
    )

    col3.metric(
        "Lines",
        stats["lines"]
    )

    col4.metric(
        "Reading Time",
        f'{stats["reading_time"]} min'
    )


# ============================================================
# GEMINI CONNECTION
# ============================================================

def test_gemini():

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=(
            "Reply exactly with: "
            "SmartDoc AI connected successfully"
        )
    )

    return response.text


# ============================================================
# AI DOCUMENT ANALYSIS
# ============================================================

def analyze_document(text):

    prompt = f"""
You are SmartDoc AI, an intelligent document analysis assistant.

Carefully analyze the document provided below.

The document may contain English, Hindi, or both languages.

Return ONLY valid JSON.

Do not use markdown.
Do not use JSON code fences.
Do not include any text before or after the JSON.

Return exactly this structure:

{{
    "summary": "Clear and informative summary",
    "simple_explanation": "Explain the document simply",
    "key_points": [
        "Important point 1",
        "Important point 2",
        "Important point 3"
    ],
    "keywords": [
        "keyword 1",
        "keyword 2",
        "keyword 3"
    ],
    "document_type": "Document category",
    "action_items": [
        "Action item 1",
        "Action item 2"
    ]
}}

Rules:

1. Analyze only information present in the document.
2. Do not invent information.
3. If no action items exist, return an empty list.
4. Provide at least 3 useful key points when enough information exists.
5. Provide 5 to 10 relevant keywords when possible.
6. Keep the summary concise but informative.
7. Understand minor OCR errors when reasonably possible.
8. Write the analysis in the main language of the document.

DOCUMENT TEXT:

{text}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    response_text = response.text.strip()

    if response_text.startswith("```json"):
        response_text = response_text[7:]

    elif response_text.startswith("```"):
        response_text = response_text[3:]

    if response_text.endswith("```"):
        response_text = response_text[:-3]

    return json.loads(
        response_text.strip()
    )


# ============================================================
# REPORT
# ============================================================

def generate_ai_report(result, original_text):

    key_points = "\n".join(
        f"- {point}"
        for point in result.get("key_points", [])
    )

    keywords = ", ".join(
        result.get("keywords", [])
    )

    action_items_list = result.get(
        "action_items",
        []
    )

    if action_items_list:
        action_items = "\n".join(
            f"- {item}"
            for item in action_items_list
        )
    else:
        action_items = "No action items detected."


    return f"""
SMARTDOC AI - DOCUMENT ANALYSIS REPORT

==================================================

DOCUMENT TYPE

{result.get("document_type", "Unknown")}

==================================================

SUMMARY

{result.get("summary", "No summary available.")}

==================================================

SIMPLE EXPLANATION

{result.get("simple_explanation", "No explanation available.")}

==================================================

KEY POINTS

{key_points}

==================================================

KEYWORDS

{keywords}

==================================================

ACTION ITEMS

{action_items}

==================================================

ORIGINAL OCR TEXT

{original_text}
"""


# ============================================================
# AI ANALYSIS UI
# ============================================================

def display_ai_analysis(text):

    st.markdown("## 🤖 AI Document Analysis")

    st.caption(
        "Generate a structured understanding of the current document."
    )

    if not text.strip():
        st.warning(
            "No readable text is available for analysis."
        )
        return

    if client is None:
        st.error(
            "Gemini API is not configured. "
            "Check your .streamlit/secrets.toml file."
        )
        return


    if (
        st.session_state.analyzed_text
        and st.session_state.analyzed_text != text
    ):
        st.session_state.ai_result = None
        st.session_state.analyzed_text = ""


    if st.button(
        "✨ Analyze Document",
        type="primary",
        use_container_width=True,
        key=f"analyze_{hash(text)}"
    ):

        try:

            with st.spinner(
                "Analyzing the document..."
            ):

                result = analyze_document(text)

            st.session_state.ai_result = result
            st.session_state.analyzed_text = text

        except json.JSONDecodeError:

            st.error(
                "Gemini returned an invalid response format. "
                "Please try again."
            )

        except Exception as e:

            show_ai_error(e)


    if (
        st.session_state.ai_result is None
        or st.session_state.analyzed_text != text
    ):
        st.info(
            "Click Analyze Document to generate the summary, "
            "key points, keywords, document type, and action items."
        )
        return


    result = st.session_state.ai_result

    st.success(
        "Analysis completed successfully."
    )


    st.markdown("### 📌 Summary")

    st.write(
        result.get(
            "summary",
            "Summary unavailable."
        )
    )


    st.markdown("### 💡 Simple Explanation")

    st.write(
        result.get(
            "simple_explanation",
            "Explanation unavailable."
        )
    )


    col1, col2 = st.columns(2)


    with col1:

        st.markdown("### ⭐ Key Points")

        key_points = result.get(
            "key_points",
            []
        )

        if key_points:
            for point in key_points:
                st.markdown(f"- {point}")
        else:
            st.write(
                "No key points detected."
            )


    with col2:

        st.markdown("### 🏷️ Keywords")

        keywords = result.get(
            "keywords",
            []
        )

        if keywords:
            for keyword in keywords:
                st.markdown(f"- {keyword}")
        else:
            st.write(
                "No keywords detected."
            )


    st.markdown("### 📂 Document Type")

    st.info(
        result.get(
            "document_type",
            "Unknown"
        )
    )


    st.markdown("### ✅ Action Items")

    action_items = result.get(
        "action_items",
        []
    )

    if action_items:
        for item in action_items:
            st.markdown(f"- {item}")
    else:
        st.write(
            "No action items detected."
        )


    report = generate_ai_report(
        result,
        text
    )


    st.download_button(
        "⬇️ Download Complete AI Report",
        report,
        "smartdoc_ai_report.txt",
        mime="text/plain",
        use_container_width=True,
        key=f"report_{hash(text)}"
    )


# ============================================================
# DOCUMENT CHAT
# ============================================================

def ask_document_question(
    document_text,
    question,
    chat_history
):

    recent_history = chat_history[-6:]

    history_text = ""

    for message in recent_history:

        role = message["role"].upper()
        content = message["content"]

        history_text += (
            f"\n{role}: {content}\n"
        )


    prompt = f"""
You are SmartDoc AI, a document question-answering assistant.

Answer the current question using ONLY the uploaded document.

Conversation history may only be used to understand
follow-up questions.

Rules:

1. The uploaded document is the only source of factual information.
2. Do not invent information.
3. If the answer is unavailable in the document, say:
"The answer is not available in the uploaded document."
4. Use conversation history only for conversational context.
5. Keep answers clear and useful.
6. Answer in the same language as the user's question.
7. Understand minor OCR errors when reasonably possible.

UPLOADED DOCUMENT:

{document_text}

RECENT CONVERSATION:

{history_text}

CURRENT QUESTION:

{question}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return response.text.strip()


def display_document_chat(text):

    st.markdown("## 💬 Chat With Your Document")

    st.caption(
        "Ask questions and continue a conversation grounded "
        "in the current document."
    )

    if not text.strip():
        st.warning(
            "No document text is available for chat."
        )
        return


    document_id = hash(text)


    if st.session_state.chat_document_id != document_id:

        st.session_state.chat_history = []
        st.session_state.chat_document_id = document_id


    if not st.session_state.chat_history:

        st.info(
            "Try asking: What is this document about? "
            "What are the main points? Are there any deadlines?"
        )


    for message in st.session_state.chat_history:

        with st.chat_message(
            message["role"]
        ):
            st.markdown(
                message["content"]
            )


    question = st.chat_input(
        "Ask a question about this document...",
        key=f"chat_input_{document_id}"
    )


    if question:

        if client is None:

            st.error(
                "Gemini API is not connected."
            )

            return


        with st.chat_message("user"):
            st.markdown(question)


        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question
            }
        )


        try:

            with st.chat_message("assistant"):

                with st.spinner(
                    "Reading the document..."
                ):

                    answer = ask_document_question(
                        text,
                        question,
                        st.session_state.chat_history[:-1]
                    )

                st.markdown(answer)


            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

        except Exception as e:

            show_ai_error(e)


    if st.session_state.chat_history:

        if st.button(
            "🗑️ Clear Chat History",
            use_container_width=True,
            key=f"clear_chat_{document_id}"
        ):
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# DOCUMENT WORKSPACE
# ============================================================

def display_document_workspace(
    edited_text,
    download_filename,
    preview_items=None
):

    document_tab, analysis_tab, chat_tab = st.tabs(
        [
            "📄 Document",
            "🤖 AI Analysis",
            "💬 Document Chat"
        ]
    )


    with document_tab:

        st.markdown("## 📄 Document Workspace")

        st.caption(
            "Review OCR results, correct recognition mistakes, "
            "inspect document statistics, and download the text."
        )


        if preview_items:

            with st.expander(
                "👁️ View Source Preview",
                expanded=False
            ):

                for title, image in preview_items:

                    st.markdown(
                        f"#### {title}"
                    )

                    st.image(
                        image,
                        width=500
                    )


        text_key = (
            f"workspace_text_"
            f"{hash(edited_text)}"
        )


        reviewed_text = st.text_area(
            "✏️ Review & Edit Extracted Text",
            value=edited_text,
            height=450,
            key=text_key
        )


        st.caption(
            "Changes made here are used by AI Analysis "
            "and Document Chat."
        )


        display_statistics(
            reviewed_text
        )


        st.download_button(
            "⬇️ Download Document Text",
            reviewed_text,
            download_filename,
            mime="text/plain",
            use_container_width=True,
            key=f"download_{hash(reviewed_text)}"
        )


    with analysis_tab:

        display_ai_analysis(
            reviewed_text
        )


    with chat_tab:

        display_document_chat(
            reviewed_text
        )


# ============================================================
# PRODUCT HEADER
# ============================================================

st.title("📄 SmartDoc AI")

st.markdown(
    "### Turn documents into clear, useful information."
)

st.write(
    "Extract Hindi and English text from images, PDFs, "
    "or camera captures. Review the OCR result, generate "
    "structured AI analysis, and chat with your document."
)

header_col1, header_col2, header_col3 = st.columns(3)

header_col1.info(
    "🔍 Hindi + English OCR"
)

header_col2.info(
    "✨ Structured AI Analysis"
)

header_col3.info(
    "💬 Document-Grounded Chat"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📄 SmartDoc AI")

st.sidebar.caption(
    "AI-powered document understanding workspace"
)


st.sidebar.markdown("### 📥 Input Source")


option = st.sidebar.radio(
    "Choose how to add a document",
    [
        "Upload Images",
        "Upload PDF",
        "Camera OCR"
    ]
)


st.sidebar.divider()


st.sidebar.markdown("### 🤖 AI Status")


if client is not None:

    st.sidebar.success(
        "Gemini API configured"
    )

else:

    st.sidebar.error(
        "Gemini API not configured"
    )


if st.sidebar.button(
    "Test AI Connection",
    use_container_width=True
):

    if client is None:

        st.sidebar.error(
            "Gemini API key not found."
        )

    else:

        try:

            with st.sidebar:

                with st.spinner(
                    "Testing connection..."
                ):

                    result = test_gemini()

                st.success(result)

        except Exception as e:

            show_ai_error(e)


st.sidebar.divider()


st.sidebar.markdown("### 🔄 Workspace")


if st.sidebar.button(
    "Clear Current Document",
    use_container_width=True
):

    reset_app()

    st.rerun()


st.sidebar.caption(
    f"Maximum file size: {MAX_FILE_SIZE_MB} MB"
)

st.sidebar.caption(
    "Supported OCR languages: English + Hindi"
)


# ============================================================
# IMAGE INPUT
# ============================================================

if option == "Upload Images":

    st.markdown("## 🖼️ Upload Images")

    st.caption(
        "Upload up to 10 PNG, JPG, or JPEG images."
    )


    files = st.file_uploader(
        "Choose image files",
        type=[
            "png",
            "jpg",
            "jpeg"
        ],
        accept_multiple_files=True,
        key=(
            f"image_uploader_"
            f"{st.session_state.uploader_version}"
        )
    )


    if not files:

        st.info(
            "Upload one or more images to start OCR processing."
        )


    else:

        if len(files) > 10:

            st.error(
                "Maximum 10 images allowed."
            )

            st.stop()


        for file in files:

            if not is_file_size_valid(file):

                st.error(
                    f"{file.name} exceeds the "
                    f"{MAX_FILE_SIZE_MB} MB limit."
                )

                st.stop()


        combined_text = ""

        preview_items = []


        with st.spinner(
            "Extracting text from images..."
        ):

            for i, file in enumerate(files):

                try:

                    image = Image.open(
                        file
                    ).convert("RGB")


                    text = run_ocr(
                        image
                    )


                except Exception as e:

                    st.error(
                        f"Could not process {file.name}: {e}"
                    )

                    st.stop()


                preview_items.append(
                    (
                        f"Image {i + 1}: {file.name}",
                        image
                    )
                )


                combined_text += (
                    f"\n\n"
                    f"===== IMAGE {i + 1} ====="
                    f"\n\n{text}"
                )


        st.success(
            f"OCR completed for {len(files)} image(s)."
        )


        display_document_workspace(
            combined_text.strip(),
            "ocr_result.txt",
            preview_items
        )


# ============================================================
# PDF INPUT
# ============================================================

elif option == "Upload PDF":

    st.markdown("## 📕 Upload PDF")

    st.caption(
        "Upload a PDF document for OCR extraction and analysis."
    )


    pdf_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        key=(
            f"pdf_uploader_"
            f"{st.session_state.uploader_version}"
        )
    )


    if not pdf_file:

        st.info(
            "Upload a PDF to start OCR processing."
        )


    else:

        if not is_file_size_valid(pdf_file):

            st.error(
                f"The PDF exceeds the "
                f"{MAX_FILE_SIZE_MB} MB limit."
            )

            st.stop()


        try:

            with st.spinner(
                "Converting and processing PDF pages..."
            ):

                pages = convert_from_bytes(
                    pdf_file.read()
                )


                combined_text = ""

                preview_items = []


                for i, page in enumerate(pages):

                    text = run_ocr(
                        page
                    )


                    preview_items.append(
                        (
                            f"Page {i + 1}",
                            page
                        )
                    )


                    combined_text += (
                        f"\n\n"
                        f"===== PAGE {i + 1} ====="
                        f"\n\n{text}"
                    )


            st.success(
                f"OCR completed for {len(pages)} page(s)."
            )


            display_document_workspace(
                combined_text.strip(),
                "pdf_ocr_result.txt",
                preview_items
            )


        except Exception as e:

            st.error(
                f"PDF Processing Error: {e}"
            )


# ============================================================
# CAMERA INPUT
# ============================================================

elif option == "Camera OCR":

    st.markdown("## 📷 Camera OCR")

    st.caption(
        "Capture a document image using your camera."
    )


    camera = st.camera_input(
        "Take a picture",
        key=(
            f"camera_"
            f"{st.session_state.uploader_version}"
        )
    )


    if not camera:

        st.info(
            "Capture a document image to start OCR processing."
        )


    else:

        try:

            image = Image.open(
                camera
            ).convert("RGB")


            with st.spinner(
                "Extracting text from camera image..."
            ):

                text = run_ocr(
                    image
                )


            st.success(
                "OCR completed successfully."
            )


            display_document_workspace(
                text,
                "camera_ocr.txt",
                [
                    (
                        "Camera Capture",
                        image
                    )
                ]
            )


        except Exception as e:

            st.error(
                f"Camera OCR Processing Error: {e}"
            )
