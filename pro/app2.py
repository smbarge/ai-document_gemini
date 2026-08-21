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

DOCUMENT_PATH = "../docs/12thMarkSheets/HSC_2.jpeg"

OUTPUT_FILE = "output/12th_marksheet_result.json"


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

    logger.info("Document AI client initialized")

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

    logger.info("Starting Enterprise Document AI")

    client = create_document_ai_client()

    processor_name = create_processor_name()

    logger.info("Processor: %s", processor_name)

    mime_type = get_mime_type(document_path)

    logger.info("MIME type: %s", mime_type)

    with open(document_path, "rb") as file:
        document_content = file.read()

    logger.info(
        "Document loaded: %.2f MB",
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

    logger.info("Sending 12th marksheet to Enterprise Document AI...")

    start_time = time.perf_counter()

    result = client.process_document(
        request=request
    )

    processing_time = time.perf_counter() - start_time

    logger.info(
        "Enterprise Document AI completed in %.3f seconds",
        processing_time
    )

    ocr_text = result.document.text

    logger.info(
        "OCR text extracted: %d characters",
        len(ocr_text)
    )

    return ocr_text


# ============================================================
# GEMINI 2.5 PRO EXTRACTION
# ============================================================

def extract_with_gemini(ocr_text):

    logger.info("Starting Gemini 2.5 Pro extraction")

    prompt = f"""
You are a document data extraction system.

The following OCR text belongs to a 12th standard HSC marksheet.

Extract the information from the marksheet and return the result
as JSON.

Extract these fields:

- candidate_name
- seat_number
- mother_name
- percentage
- passing_year
- total_marks
- obtained_marks
- stream

Rules:

1. Use only information available in the OCR text.
2. Do not invent information.
3. Do not guess missing information.
4. Preserve names exactly as they appear.
5. Preserve values exactly as they appear.
6. If a field is not available, return an empty string.
7. Return only JSON.
8. Do not add explanations.
9. Do not add markdown.
10. Do not add extra fields.

OCR TEXT:

--------------------------------
{ocr_text}
--------------------------------
"""

    logger.info("Initializing Gemini Vertex AI client...")

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    logger.info("Gemini client initialized")

    logger.info("Sending OCR text to Gemini 2.5 Pro...")

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt
    )

    processing_time = time.perf_counter() - start_time

    logger.info(
        "Gemini completed in %.3f seconds",
        processing_time
    )

    result = json.loads(response.text)

    logger.info("Gemini JSON extraction completed")

    return result


# ============================================================
# SAVE JSON
# ============================================================

def save_json(result):

    output_directory = os.path.dirname(OUTPUT_FILE)

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True
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
        "JSON saved: %s",
        os.path.abspath(OUTPUT_FILE)
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 60)
    logger.info("12TH MARKSHEET DATA EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info("Project ID       : %s", PROJECT_ID)
    logger.info("Location         : %s", LOCATION)
    logger.info("Processor ID     : %s", PROCESSOR_ID)
    logger.info("Processor Version: %s", PROCESSOR_VERSION)
    logger.info("Gemini Model     : %s", GEMINI_MODEL)
    logger.info("Input            : %s", DOCUMENT_PATH)
    logger.info("Output           : %s", OUTPUT_FILE)

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

    final_result = extract_with_gemini(
        ocr_text
    )

    # ========================================================
    # STEP 3 - SAVE JSON
    # ========================================================

    logger.info("")
    logger.info("=" * 60)
    logger.info("STEP 3 - SAVE JSON")
    logger.info("=" * 60)

    save_json(
        final_result
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
            final_result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 60)

    print("\nJSON FILE:")
    print(os.path.abspath(OUTPUT_FILE))

    total_time = (
        time.perf_counter()
        - application_start
    )

    logger.info(
        "TOTAL EXECUTION TIME: %.3f seconds",
        total_time
    )

    logger.info("=" * 60)
    logger.info("12TH MARKSHEET DATA EXTRACTION FINISHED")
    logger.info("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()