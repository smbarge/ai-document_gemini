import json
import logging
import os
import time

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google import genai
from google.genai.types import HttpOptions


# ============================================================
# GOOGLE CLOUD CONFIGURATION
# ============================================================

PROJECT_ID = "document-ai-test-506006"

LOCATION = "asia-south1"

PROCESSOR_ID = "ec631cf67f66f191"

PROCESSOR_VERSION = "pretrained-foundation-model-v1.5-2025-08-06"


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-pro"


# ============================================================
# FILE CONFIGURATION
# ============================================================

DOCUMENT_PATH = "../docs/aadharCards/Adhar1.png"

OUTPUT_FILE = "output/aadhar_result.json"


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
# GET MIME TYPE
# ============================================================

def get_mime_type(document_path):

    extension = os.path.splitext(document_path)[1].lower()

    if extension == ".pdf":
        return "application/pdf"

    if extension in [".jpg", ".jpeg"]:
        return "image/jpeg"

    if extension == ".png":
        return "image/png"

    if extension in [".tif", ".tiff"]:
        return "image/tiff"

    return "application/octet-stream"


# ============================================================
# CREATE DOCUMENT AI CLIENT
# ============================================================

def create_document_ai_client():

    logger.info("Initializing Enterprise Document AI client...")

    client_options = ClientOptions(
        api_endpoint=f"{LOCATION}-documentai.googleapis.com"
    )

    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )

    logger.info("Enterprise Document AI client initialized")

    return client


# ============================================================
# CREATE PROCESSOR NAME
# ============================================================

def create_processor_name():

    return (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )


# ============================================================
# ENTERPRISE DOCUMENT AI OCR
# ============================================================

def extract_ocr_text(document_path):

    logger.info("")
    logger.info("=" * 60)
    logger.info("ENTERPRISE DOCUMENT AI OCR STARTED")
    logger.info("=" * 60)

    client = create_document_ai_client()

    processor_name = create_processor_name()

    logger.info(
        "Processor: %s",
        processor_name
    )

    mime_type = get_mime_type(
        document_path
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    with open(
        document_path,
        "rb"
    ) as file:

        document_content = file.read()

    logger.info(
        "Document size: %.2f MB",
        len(document_content) / (1024 * 1024)
    )

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )

    logger.info(
        "Sending Aadhaar document to Enterprise Document AI..."
    )

    start_time = time.perf_counter()

    response = client.process_document(
        request=request
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Enterprise OCR completed in %.3f seconds",
        processing_time
    )

    ocr_text = response.document.text

    logger.info(
        "OCR text extracted: %d characters",
        len(ocr_text)
    )

    logger.info("=" * 60)
    logger.info("ENTERPRISE DOCUMENT AI OCR FINISHED")
    logger.info("=" * 60)

    return ocr_text


# ============================================================
# GEMINI 2.5 PRO EXTRACTION
# ============================================================

def extract_with_gemini(ocr_text):

    logger.info("")
    logger.info("=" * 60)
    logger.info("GEMINI 2.5 PRO EXTRACTION STARTED")
    logger.info("=" * 60)

    prompt = f"""
You are an expert document data extraction system.

The following OCR text belongs to an Aadhaar document.

Extract the following information:

1. full_name
2. aadhaar_number
3. date_of_birth
4. gender
5. address

Instructions:

- Use only information present in the OCR text.
- Do not invent information.
- Do not guess missing information.
- Do not calculate anything.
- Do not modify names.
- Preserve the Aadhaar number exactly as it appears.
- Preserve the date of birth exactly as it appears.
- Preserve the gender exactly as it appears.
- Preserve the complete address.
- If a field is not available, return an empty string.
- Return only JSON.
- Do not return markdown.
- Do not return explanations.
- Do not add extra fields.

Return JSON in exactly this structure:

{{
    "full_name": "",
    "aadhaar_number": "",
    "date_of_birth": "",
    "gender": "",
    "address": ""
}}

OCR TEXT:

----------------------------------------
{ocr_text}
----------------------------------------
"""

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

    logger.info(
        "Sending OCR text to Gemini 2.5 Pro..."
    )

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "Gemini 2.5 Pro completed in %.3f seconds",
        processing_time
    )

    logger.info(
        "Gemini response received"
    )

    return response.text


# ============================================================
# SAVE JSON
# ============================================================

def save_json(gemini_result):

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING JSON RESULT")
    logger.info("=" * 60)

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
        )

    result = json.loads(
        gemini_result
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False
        )

    logger.info(
        "JSON saved successfully"
    )

    logger.info(
        "Output file: %s",
        os.path.abspath(OUTPUT_FILE)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 60)
    logger.info("AADHAAR DATA EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info(
        "Project ID       : %s",
        PROJECT_ID
    )

    logger.info(
        "Location         : %s",
        LOCATION
    )

    logger.info(
        "Processor ID     : %s",
        PROCESSOR_ID
    )

    logger.info(
        "Processor Version: %s",
        PROCESSOR_VERSION
    )

    logger.info(
        "Gemini Model     : %s",
        GEMINI_MODEL
    )

    logger.info(
        "Input File       : %s",
        DOCUMENT_PATH
    )

    logger.info(
        "Output File      : %s",
        OUTPUT_FILE
    )

    # ========================================================
    # STEP 1 - ENTERPRISE DOCUMENT AI
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 1 - ENTERPRISE DOCUMENT AI")
    logger.info("=" * 60)

    ocr_text = extract_ocr_text(
        DOCUMENT_PATH
    )

    print("\n")
    print("=" * 60)
    print("OCR TEXT")
    print("=" * 60)
    print(ocr_text)
    print("=" * 60)

    # ========================================================
    # STEP 2 - GEMINI 2.5 PRO
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 2 - GEMINI 2.5 PRO")
    logger.info("=" * 60)

    gemini_result = extract_with_gemini(
        ocr_text
    )

    print("\n")
    print("=" * 60)
    print("GEMINI RESULT")
    print("=" * 60)
    print(gemini_result)
    print("=" * 60)

    # ========================================================
    # STEP 3 - SAVE JSON
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 - SAVE JSON")
    logger.info("=" * 60)

    save_json(
        gemini_result
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print("\n")
    print("=" * 60)
    print("FINAL JSON OUTPUT")
    print("=" * 60)

    print(
        json.dumps(
            json.loads(gemini_result),
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 60)

    print("\nJSON FILE:")
    print(
        os.path.abspath(
            OUTPUT_FILE
        )
    )

    total_time = (
        time.perf_counter()
        - application_start
    )

    logger.info(
        "TOTAL EXECUTION TIME: %.3f seconds",
        total_time
    )

    logger.info("=" * 60)
    logger.info("AADHAAR DATA EXTRACTION FINISHED")
    logger.info("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()