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

PROCESSOR_VERSION = (
    "pretrained-foundation-model-v1.5-2025-08-06"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash-lite"

GEMINI_LOCATION = "global"


# ============================================================
# FILE CONFIGURATION
# ============================================================

DOCUMENT_PATH = "../docs/10thMarkSheets/SSC_1.jpeg"

OUTPUT_FILE = "output/10th_marksheet_result.json"


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

    extension = os.path.splitext(
        document_path
    )[1].lower()

    mime_types = {
        ".pdf": "application/pdf",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
    }

    if extension not in mime_types:
        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return mime_types[extension]


# ============================================================
# ENTERPRISE DOCUMENT AI OCR
# ============================================================

def run_document_ai_ocr(document_path):

    total_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ENTERPRISE DOCUMENT AI OCR STARTED")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Configuration logs
    # --------------------------------------------------------

    logger.info(
        "Project ID: %s",
        PROJECT_ID
    )

    logger.info(
        "Location: %s",
        LOCATION
    )

    logger.info(
        "Processor ID: %s",
        PROCESSOR_ID
    )

    logger.info(
        "Processor Version: %s",
        PROCESSOR_VERSION
    )

    logger.info(
        "Input file: %s",
        document_path
    )

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not os.path.isfile(document_path):

        raise FileNotFoundError(
            f"Document not found: "
            f"{os.path.abspath(document_path)}"
        )

    logger.info(
        "Document found successfully"
    )

    # --------------------------------------------------------
    # MIME type
    # --------------------------------------------------------

    mime_type = get_mime_type(
        document_path
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # --------------------------------------------------------
    # Create Document AI client
    # --------------------------------------------------------

    client_start = time.perf_counter()

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

    client_time = (
        time.perf_counter()
        - client_start
    )

    logger.info(
        "Document AI client initialized in %.3f seconds",
        client_time
    )

    # --------------------------------------------------------
    # Processor resource
    # --------------------------------------------------------

    processor_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        "Processor resource: %s",
        processor_name
    )

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    read_start = time.perf_counter()

    with open(
        document_path,
        "rb"
    ) as file:

        document_content = file.read()

    read_time = (
        time.perf_counter()
        - read_start
    )

    logger.info(
        "Document read time: %.3f seconds",
        read_time
    )

    logger.info(
        "Document size: %.3f MB",
        len(document_content) / (1024 * 1024)
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
        name=processor_name,
        raw_document=raw_document,
        process_options=process_options
    )

    # --------------------------------------------------------
    # Send to Enterprise Document AI
    # --------------------------------------------------------

    logger.info("")
    logger.info(
        "Sending 10th marksheet to Enterprise Document AI..."
    )

    ocr_start = time.perf_counter()

    response = client.process_document(
        request=request
    )

    ocr_time = (
        time.perf_counter()
        - ocr_start
    )

    logger.info(
        "Enterprise Document AI response received"
    )

    logger.info(
        "Enterprise OCR API time: %.3f seconds",
        ocr_time
    )

    # --------------------------------------------------------
    # OCR result
    # --------------------------------------------------------

    document = response.document

    ocr_text = document.text

    page_count = len(
        document.pages
    )

    logger.info(
        "Pages detected: %d",
        page_count
    )

    logger.info(
        "OCR text length: %d characters",
        len(ocr_text)
    )

    # --------------------------------------------------------
    # Display OCR text
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("ENTERPRISE OCR TEXT")
    print("=" * 70)

    print(ocr_text)

    print("=" * 70)

    total_ocr_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        "TOTAL ENTERPRISE OCR TIME: %.3f seconds",
        total_ocr_time
    )

    logger.info("=" * 70)
    logger.info("ENTERPRISE DOCUMENT AI OCR FINISHED")
    logger.info("=" * 70)

    return ocr_text


# ============================================================
# GEMINI 2.5 FLASH-LITE
# ============================================================

def extract_with_gemini(ocr_text):

    total_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info("GEMINI 2.5 FLASH-LITE EXTRACTION STARTED")
    logger.info("=" * 70)

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "OCR text length sent to Gemini: %d characters",
        len(ocr_text)
    )

    # --------------------------------------------------------
    # Structured output schema
    # --------------------------------------------------------

    response_schema = {
        "type": "OBJECT",

        "properties": {

            "candidate_name": {
                "type": "STRING"
            },

            "seat_number": {
                "type": "STRING"
            },

            "mother_name": {
                "type": "STRING"
            },

            "father_name": {
                "type": "STRING"
            },

            "date_of_birth": {
                "type": "STRING"
            },

            "school_name": {
                "type": "STRING"
            },

            "examination_year": {
                "type": "STRING"
            },

            "passing_year": {
                "type": "STRING"
            },

            "total_marks": {
                "type": "STRING"
            },

            "obtained_marks": {
                "type": "STRING"
            },

            "percentage": {
                "type": "STRING"
            },

            "grade": {
                "type": "STRING"
            },

            "result": {
                "type": "STRING"
            }
        },

        "required": [
            "candidate_name",
            "seat_number",
            "mother_name",
            "father_name",
            "date_of_birth",
            "school_name",
            "examination_year",
            "passing_year",
            "total_marks",
            "obtained_marks",
            "percentage",
            "grade",
            "result"
        ]
    }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an enterprise document understanding system.

The following text was extracted using
Google Cloud Enterprise Document AI OCR.

The document is a 10th standard SSC marksheet.

Your task is to extract the following information:

1. candidate_name
2. seat_number
3. mother_name
4. father_name
5. date_of_birth
6. school_name
7. examination_year
8. passing_year
9. total_marks
10. obtained_marks
11. percentage
12. grade
13. result

IMPORTANT INSTRUCTIONS:

- Extract information only from the OCR text.
- Do not invent any information.
- Do not guess missing values.
- Do not calculate percentage.
- Do not calculate marks.
- Do not modify names.
- Preserve names exactly as they appear.
- Preserve seat numbers exactly as they appear.
- Preserve dates as they appear.
- Preserve marks exactly as written.
- If a field is not available in the OCR text,
  return an empty string.
- Return only the requested JSON structure.

OCR TEXT:

--------------------------------------------------
{ocr_text}
--------------------------------------------------
"""

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    logger.info(
        "Initializing Gemini Vertex AI client..."
    )

    client_start = time.perf_counter()

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=GEMINI_LOCATION,
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = (
        time.perf_counter()
        - client_start
    )

    logger.info(
        "Gemini client initialized in %.3f seconds",
        client_time
    )

    # --------------------------------------------------------
    # Gemini API
    # --------------------------------------------------------

    logger.info(
        "Sending OCR text to Gemini 2.5 Flash-Lite..."
    )

    gemini_start = time.perf_counter()

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
        - gemini_start
    )

    logger.info(
        "Gemini API response received in %.3f seconds",
        gemini_api_time
    )

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    result = json.loads(
        response.text
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("GEMINI 2.5 FLASH-LITE RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 70)

    total_gemini_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        "TOTAL GEMINI EXTRACTION TIME: %.3f seconds",
        total_gemini_time
    )

    logger.info("=" * 70)
    logger.info(
        "GEMINI 2.5 FLASH-LITE EXTRACTION FINISHED"
    )
    logger.info("=" * 70)

    return result


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    start_time = time.perf_counter()

    logger.info(
        "Saving final JSON..."
    )

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

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

    save_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "JSON saved successfully"
    )

    logger.info(
        "Output file: %s",
        os.path.abspath(OUTPUT_FILE)
    )

    logger.info(
        "JSON save time: %.3f seconds",
        save_time
    )

    return save_time


# ============================================================
# MAIN
# ============================================================

def main():

    application_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "10TH MARKSHEET EXTRACTION PIPELINE STARTED"
    )
    logger.info("=" * 70)

    try:

        # ====================================================
        # STEP 1 - ENTERPRISE OCR
        # ====================================================

        step1_start = time.perf_counter()

        ocr_text = run_document_ai_ocr(
            DOCUMENT_PATH
        )

        step1_time = (
            time.perf_counter()
            - step1_start
        )

        logger.info(
            "STEP 1 - Enterprise OCR: %.3f seconds",
            step1_time
        )

        # ====================================================
        # STEP 2 - GEMINI
        # ====================================================

        step2_start = time.perf_counter()

        final_result = extract_with_gemini(
            ocr_text
        )

        step2_time = (
            time.perf_counter()
            - step2_start
        )

        logger.info(
            "STEP 2 - Gemini 2.5 Flash-Lite: %.3f seconds",
            step2_time
        )

        # ====================================================
        # STEP 3 - SAVE
        # ====================================================

        step3_start = time.perf_counter()

        save_time = save_result(
            final_result
        )

        step3_time = (
            time.perf_counter()
            - step3_start
        )

        logger.info(
            "STEP 3 - Save JSON: %.3f seconds",
            step3_time
        )

        # ====================================================
        # TOTAL PIPELINE TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - application_start
        )

        # ====================================================
        # FINAL PERFORMANCE SUMMARY
        # ====================================================

        logger.info("")
        logger.info("=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)

        logger.info(
            "Enterprise OCR Time       : %.3f seconds",
            step1_time
        )

        logger.info(
            "Gemini Flash-Lite Time    : %.3f seconds",
            step2_time
        )

        logger.info(
            "JSON Save Time            : %.3f seconds",
            step3_time
        )

        logger.info(
            "-" * 70
        )

        logger.info(
            "TOTAL PIPELINE TIME       : %.3f seconds",
            total_time
        )

        logger.info(
            "TOTAL PIPELINE TIME       : %.2f minutes",
            total_time / 60
        )

        logger.info("=" * 70)

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print("")
        print("=" * 70)
        print("FINAL 10TH MARKSHEET JSON")
        print("=" * 70)

        print(
            json.dumps(
                final_result,
                indent=4,
                ensure_ascii=False
            )
        )

        print("=" * 70)

        print(
            f"Saved to: {os.path.abspath(OUTPUT_FILE)}"
        )

        print("=" * 70)

        logger.info(
            "10TH MARKSHEET EXTRACTION FINISHED SUCCESSFULLY"
        )

    except Exception as error:

        logger.exception(
            "10th marksheet extraction failed: %s",
            error
        )

        raise


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()