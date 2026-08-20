import os
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai

from google import genai
from google.genai.types import HttpOptions


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

LOCATION = os.getenv(
    "GOOGLE_CLOUD_LOCATION",
    "us"
)

PROCESSOR_ID = os.getenv(
    "DOCUMENT_AI_PROCESSOR_ID"
)

PROCESSOR_VERSION = os.getenv(
    "DOCUMENT_AI_PROCESSOR_VERSION",
    "stable"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-pro"
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "docs/10thMarkSheets/ssc_1.jpeg"

OUTPUT_FILE = "output/marksheet_result.json"


# ============================================================
# MIME TYPE
# ============================================================

def get_mime_type(file_path):

    extension = Path(
        file_path
    ).suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    return mime_types.get(extension)


# ============================================================
# DOCUMENT AI OCR
# ============================================================

def run_enterprise_ocr(file_path):

    # Start OCR timer
    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("ENTERPRISE DOCUMENT OCR STARTED")
    logger.info("------------------------------------------")

    logger.info(
        f"Input file: {file_path}"
    )

    # --------------------------------------------------------
    # File information
    # --------------------------------------------------------

    file_path_object = Path(file_path)

    if file_path_object.exists():

        file_size_bytes = file_path_object.stat().st_size

        file_size_mb = (
            file_size_bytes / (1024 * 1024)
        )

        logger.info(
            f"File size: {file_size_mb:.3f} MB"
        )

    # --------------------------------------------------------
    # MIME type
    # --------------------------------------------------------

    mime_type = get_mime_type(
        file_path
    )

    if not mime_type:
        raise ValueError(
            f"Unsupported file type: {file_path}"
        )

    logger.info(
        f"MIME type: {mime_type}"
    )

    logger.info(
        f"Processor ID: {PROCESSOR_ID}"
    )

    # --------------------------------------------------------
    # Document AI endpoint
    # --------------------------------------------------------

    client_options = ClientOptions(
        api_endpoint=(
            f"{LOCATION}-documentai.googleapis.com"
        )
    )

    client = (
        documentai.DocumentProcessorServiceClient(
            client_options=client_options
        )
    )

    logger.info(
        "Document AI client initialized"
    )

    # --------------------------------------------------------
    # Processor version resource
    # --------------------------------------------------------

    processor_version_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        f"Processor version: {PROCESSOR_VERSION}"
    )

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    logger.info(
        "Reading input document..."
    )

    with open(
        file_path,
        "rb"
    ) as file:

        document_content = file.read()

    logger.info(
        "Input document read successfully"
    )

    # --------------------------------------------------------
    # Raw document
    # --------------------------------------------------------

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    # --------------------------------------------------------
    # OCR configuration
    # --------------------------------------------------------

    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=True,
            enable_symbol=True
        )
    )

    # --------------------------------------------------------
    # Request
    # --------------------------------------------------------

    request = documentai.ProcessRequest(
        name=processor_version_name,
        raw_document=raw_document,
        process_options=process_options
    )

    # --------------------------------------------------------
    # Send document to Enterprise OCR
    # --------------------------------------------------------

    logger.info(
        "Sending document to Enterprise Document OCR..."
    )

    ocr_api_start = time.perf_counter()

    response = client.process_document(
        request=request
    )

    ocr_api_time = (
        time.perf_counter()
        - ocr_api_start
    )

    logger.info(
        f"Enterprise OCR API response received "
        f"in {ocr_api_time:.3f} seconds"
    )

    document = response.document

    # --------------------------------------------------------
    # OCR information
    # --------------------------------------------------------

    page_count = len(
        document.pages
    )

    ocr_text_length = len(
        document.text
    )

    logger.info(
        "OCR completed successfully"
    )

    logger.info(
        f"Pages detected: {page_count}"
    )

    logger.info(
        f"OCR text length: {ocr_text_length} characters"
    )

    # --------------------------------------------------------
    # Print OCR text
    # --------------------------------------------------------

    print(
        "\n========== OCR TEXT ==========\n"
    )

    print(
        document.text
    )

    # --------------------------------------------------------
    # Total OCR time
    # --------------------------------------------------------

    total_ocr_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        f"TOTAL ENTERPRISE OCR TIME: "
        f"{total_ocr_time:.3f} seconds"
    )

    logger.info("------------------------------------------")
    logger.info("ENTERPRISE DOCUMENT OCR FINISHED")
    logger.info("------------------------------------------")

    return document


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_with_gemini(ocr_document):

    # Start Gemini timer
    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("GEMINI 2.5 PRO EXTRACTION STARTED")
    logger.info("------------------------------------------")

    logger.info(
        f"Gemini model: {GEMINI_MODEL}"
    )

    # --------------------------------------------------------
    # Complete OCR text
    # --------------------------------------------------------

    ocr_text = ocr_document.text

    logger.info(
        f"OCR text sent to Gemini: "
        f"{len(ocr_text)} characters"
    )

    # --------------------------------------------------------
    # Structured output schema
    # --------------------------------------------------------

    response_schema = {

        "type": "OBJECT",

        "properties": {

            "candidate_name": {
                "type": "STRING",
                "description": (
                    "Full name of the student/candidate "
                    "as written on the 10th marksheet."
                )
            },

            "seat_number": {
                "type": "STRING",
                "description": (
                    "Seat number, examination number, "
                    "or roll number of the student."
                )
            },

            "mother_name": {
                "type": "STRING",
                "description": (
                    "Mother's name exactly as written "
                    "on the marksheet."
                )
            },

            "percentage": {
                "type": "STRING",
                "description": (
                    "Overall percentage obtained by "
                    "the student."
                )
            },

            "passing_year": {
                "type": "STRING",
                "description": (
                    "Year in which the student passed "
                    "the examination."
                )
            }
        },

        "required": [
            "candidate_name",
            "seat_number",
            "mother_name",
            "percentage",
            "passing_year"
        ]
    }

    # --------------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document understanding system.

The following text was extracted from a 10th standard
marksheet using Google Cloud Enterprise Document OCR.

Your task is to identify the requested fields from the
OCR text.

Fields to extract:

1. candidate_name
2. seat_number
3. mother_name
4. percentage
5. passing_year

Important instructions:

- Use only information present in the OCR text.
- Understand the relationship between labels and values.
- Do not invent values.
- Do not calculate values.
- Do not correct or modify names.
- Preserve the value as written in the document.
- If a requested field cannot be identified, return an
  empty string for that field.
- Return only the requested structured fields.

OCR TEXT:

------------------------------
{ocr_text}
------------------------------
"""

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    logger.info(
        "Initializing Gemini Vertex AI client..."
    )

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    logger.info(
        "Gemini client initialized"
    )

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    logger.info(
        "Sending OCR text to Gemini 2.5 Pro..."
    )

    gemini_api_start = time.perf_counter()

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config={

            "response_mime_type": "application/json",

            "response_schema": response_schema,

            "temperature": 0
        }
    )

    gemini_api_time = (
        time.perf_counter()
        - gemini_api_start
    )

    logger.info(
        f"Gemini API response received in "
        f"{gemini_api_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # Gemini result
    # --------------------------------------------------------

    print(
        "\n========== GEMINI RESULT ==========\n"
    )

    print(
        response.text
    )

    # --------------------------------------------------------
    # Total Gemini time
    # --------------------------------------------------------

    total_gemini_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        f"TOTAL GEMINI EXTRACTION TIME: "
        f"{total_gemini_time:.3f} seconds"
    )

    logger.info("------------------------------------------")
    logger.info("GEMINI 2.5 PRO EXTRACTION FINISHED")
    logger.info("------------------------------------------")

    return response.text


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    # Start save timer
    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("SAVING RESULT")
    logger.info("------------------------------------------")

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        f"Output file: {output_path}"
    )

    # --------------------------------------------------------
    # Convert Gemini JSON
    # --------------------------------------------------------

    logger.info(
        "Converting Gemini response to JSON..."
    )

    parsed_result = json.loads(
        result
    )

    # --------------------------------------------------------
    # Write JSON
    # --------------------------------------------------------

    logger.info(
        "Writing JSON result..."
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            parsed_result,
            file,
            indent=4,
            ensure_ascii=False
        )

    # --------------------------------------------------------
    # Save time
    # --------------------------------------------------------

    save_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        f"RESULT SAVED SUCCESSFULLY"
    )

    logger.info(
        f"JSON SAVE TIME: "
        f"{save_time:.3f} seconds"
    )

    logger.info(
        f"Output file: {output_path}"
    )

    logger.info("------------------------------------------")
    logger.info("RESULT SAVING FINISHED")
    logger.info("------------------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # TOTAL APPLICATION TIMER
    # ========================================================

    application_start_time = (
        time.perf_counter()
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("10TH MARKSHEET DATA EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info(
        f"Project       : {PROJECT_ID}"
    )

    logger.info(
        f"Location      : {LOCATION}"
    )

    logger.info(
        f"Processor ID  : {PROCESSOR_ID}"
    )

    logger.info(
        f"Processor Ver : {PROCESSOR_VERSION}"
    )

    logger.info(
        f"Gemini Model  : {GEMINI_MODEL}"
    )

    logger.info(
        f"Input         : {INPUT_FILE}"
    )

    logger.info(
        f"Output        : {OUTPUT_FILE}"
    )

    # ========================================================
    # STEP 1 - ENTERPRISE OCR
    # ========================================================

    step1_start = time.perf_counter()

    document = run_enterprise_ocr(
        INPUT_FILE
    )

    step1_time = (
        time.perf_counter()
        - step1_start
    )

    logger.info(
        f"STEP 1 - Enterprise OCR: "
        f"{step1_time:.3f} seconds"
    )

    # ========================================================
    # STEP 2 - GEMINI
    # ========================================================

    step2_start = time.perf_counter()

    result = extract_with_gemini(
        document
    )

    step2_time = (
        time.perf_counter()
        - step2_start
    )

    logger.info(
        f"STEP 2 - Gemini 2.5 Pro: "
        f"{step2_time:.3f} seconds"
    )

    # ========================================================
    # STEP 3 - SAVE RESULT
    # ========================================================

    step3_start = time.perf_counter()

    save_result(
        result
    )

    step3_time = (
        time.perf_counter()
        - step3_start
    )

    logger.info(
        f"STEP 3 - Save JSON: "
        f"{step3_time:.3f} seconds"
    )

    # ========================================================
    # TOTAL EXECUTION TIME
    # ========================================================

    total_execution_time = (
        time.perf_counter()
        - application_start_time
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    logger.info(
        f"Enterprise OCR Time : "
        f"{step1_time:.3f} seconds"
    )

    logger.info(
        f"Gemini 2.5 Pro Time : "
        f"{step2_time:.3f} seconds"
    )

    logger.info(
        f"JSON Save Time      : "
        f"{step3_time:.3f} seconds"
    )

    logger.info(
        f"TOTAL TIME          : "
        f"{total_execution_time:.3f} seconds"
    )

    logger.info("=" * 60)

    logger.info(
        "10TH MARKSHEET DATA EXTRACTION FINISHED"
    )

    logger.info("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logger.exception(
            f"APPLICATION FAILED: {error}"
        )

        raise