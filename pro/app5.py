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

PROJECT_ID = os.getenv(
    "GOOGLE_CLOUD_PROJECT"
)

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

INPUT_FILE = "docs/casteCertificates/casteCertificate_1.jpg"

OUTPUT_FILE = "output/caste_certificate_result.json"


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

    return mime_types.get(
        extension
    )


# ============================================================
# ENTERPRISE DOCUMENT OCR
# ============================================================

def run_enterprise_ocr(file_path):

    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("ENTERPRISE DOCUMENT OCR STARTED")
    logger.info("CASTE CERTIFICATE")
    logger.info("------------------------------------------")

    logger.info(
        f"Input file: {file_path}"
    )

    # --------------------------------------------------------
    # FILE INFORMATION
    # --------------------------------------------------------

    file_path_object = Path(
        file_path
    )

    if file_path_object.exists():

        file_size_bytes = (
            file_path_object.stat().st_size
        )

        file_size_mb = (
            file_size_bytes
            / (1024 * 1024)
        )

        logger.info(
            f"File size: {file_size_mb:.3f} MB"
        )

    # --------------------------------------------------------
    # MIME TYPE
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
    # DOCUMENT AI CLIENT
    # --------------------------------------------------------

    client_start_time = (
        time.perf_counter()
    )

    logger.info(
        "Initializing Document AI client..."
    )

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
        - client_start_time
    )

    logger.info(
        "Document AI client initialized"
    )

    logger.info(
        f"Client initialization time: "
        f"{client_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # PROCESSOR VERSION
    # --------------------------------------------------------

    processor_version_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        f"Processor version: "
        f"{PROCESSOR_VERSION}"
    )

    # --------------------------------------------------------
    # READ DOCUMENT
    # --------------------------------------------------------

    read_start_time = (
        time.perf_counter()
    )

    logger.info(
        "Reading caste certificate document..."
    )

    with open(
        file_path,
        "rb"
    ) as file:

        document_content = (
            file.read()
        )

    read_time = (
        time.perf_counter()
        - read_start_time
    )

    logger.info(
        "Caste certificate document "
        "read successfully"
    )

    logger.info(
        f"Document size: "
        f"{len(document_content)} bytes"
    )

    logger.info(
        f"Document read time: "
        f"{read_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # RAW DOCUMENT
    # --------------------------------------------------------

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    # --------------------------------------------------------
    # OCR CONFIGURATION
    # --------------------------------------------------------

    process_options = (
        documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                enable_native_pdf_parsing=True,
                enable_image_quality_scores=True,
                enable_symbol=True
            )
        )
    )

    # --------------------------------------------------------
    # PROCESS REQUEST
    # --------------------------------------------------------

    request = documentai.ProcessRequest(
        name=processor_version_name,
        raw_document=raw_document,
        process_options=process_options
    )

    # --------------------------------------------------------
    # ENTERPRISE OCR API
    # --------------------------------------------------------

    logger.info(
        "Sending caste certificate to "
        "Enterprise Document OCR..."
    )

    ocr_api_start_time = (
        time.perf_counter()
    )

    response = client.process_document(
        request=request
    )

    ocr_api_time = (
        time.perf_counter()
        - ocr_api_start_time
    )

    logger.info(
        "Enterprise OCR API response received"
    )

    logger.info(
        f"Enterprise OCR API time: "
        f"{ocr_api_time:.3f} seconds"
    )

    document = response.document

    # --------------------------------------------------------
    # OCR INFORMATION
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
        f"OCR text length: "
        f"{ocr_text_length} characters"
    )

    # --------------------------------------------------------
    # PRINT OCR TEXT
    # --------------------------------------------------------

    print(
        "\n========== OCR TEXT ==========\n"
    )

    print(
        document.text
    )

    # --------------------------------------------------------
    # TOTAL OCR TIME
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
# GEMINI 2.5 PRO EXTRACTION
# ============================================================

def extract_with_gemini(
    ocr_document
):

    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("GEMINI 2.5 PRO EXTRACTION STARTED")
    logger.info("CASTE CERTIFICATE")
    logger.info("------------------------------------------")

    logger.info(
        f"Gemini model: {GEMINI_MODEL}"
    )

    # --------------------------------------------------------
    # OCR TEXT
    # --------------------------------------------------------

    ocr_text = (
        ocr_document.text
    )

    logger.info(
        f"OCR text length sent to Gemini: "
        f"{len(ocr_text)} characters"
    )

    # --------------------------------------------------------
    # STRUCTURED RESPONSE SCHEMA
    # --------------------------------------------------------

    response_schema = {

        "type": "OBJECT",

        "properties": {

            "candidate_name": {
                "type": "STRING",
                "description": (
                    "Full name of the person for whom "
                    "the caste certificate was issued."
                )
            },

            "certificate_number": {
                "type": "STRING",
                "description": (
                    "Caste certificate number or "
                    "certificate identification number."
                )
            },

            "issue_date": {
                "type": "STRING",
                "description": (
                    "Date on which the caste certificate "
                    "was issued."
                )
            },

            "caste": {
                "type": "STRING",
                "description": (
                    "Caste mentioned on the certificate."
                )
            },

            "category": {
                "type": "STRING",
                "description": (
                    "Caste category such as SC, ST, OBC, "
                    "or another category mentioned."
                )
            },

            "sub_caste": {
                "type": "STRING",
                "description": (
                    "Sub-caste or community name if "
                    "mentioned on the certificate."
                )
            },

            "father_name": {
                "type": "STRING",
                "description": (
                    "Father's name if present."
                )
            },

            "mother_name": {
                "type": "STRING",
                "description": (
                    "Mother's name if present."
                )
            },

            "district": {
                "type": "STRING",
                "description": (
                    "District mentioned on the certificate."
                )
            },

            "taluka": {
                "type": "STRING",
                "description": (
                    "Taluka or tehsil mentioned."
                )
            },

            "village": {
                "type": "STRING",
                "description": (
                    "Village mentioned on the certificate."
                )
            },

            "issuing_authority": {
                "type": "STRING",
                "description": (
                    "Authority or officer who issued "
                    "the certificate."
                )
            }
        },

        "required": [
            "candidate_name",
            "certificate_number",
            "issue_date",
            "caste",
            "category",
            "sub_caste",
            "father_name",
            "mother_name",
            "district",
            "taluka",
            "village",
            "issuing_authority"
        ]
    }

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are an expert document understanding system.

The following text was extracted from a caste certificate
using Google Cloud Enterprise Document OCR.

Extract the following fields:

1. candidate_name
2. certificate_number
3. issue_date
4. caste
5. category
6. sub_caste
7. father_name
8. mother_name
9. district
10. taluka
11. village
12. issuing_authority

IMPORTANT INSTRUCTIONS:

- Use only information present in the OCR text.
- Do not invent information.
- Do not calculate anything.
- Do not infer missing information.
- Do not correct spelling.
- Preserve names exactly as written.
- Preserve caste/category exactly as written.
- Preserve certificate number exactly as written.
- Preserve dates exactly as written.
- If a field is not present, return an empty string.
- Return only the requested structured fields.

OCR TEXT:

----------------------------------------
{ocr_text}
----------------------------------------
"""

    # --------------------------------------------------------
    # GEMINI CLIENT
    # --------------------------------------------------------

    logger.info(
        "Initializing Gemini Vertex AI client..."
    )

    client_start_time = (
        time.perf_counter()
    )

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = (
        time.perf_counter()
        - client_start_time
    )

    logger.info(
        "Gemini client initialized"
    )

    logger.info(
        f"Gemini client initialization time: "
        f"{client_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # GEMINI 2.5 PRO API
    # --------------------------------------------------------

    logger.info(
        "Sending OCR text to Gemini 2.5 Pro..."
    )

    gemini_api_start_time = (
        time.perf_counter()
    )

    response = client.models.generate_content(

        model=GEMINI_MODEL,

        contents=prompt,

        config={

            "response_mime_type":
                "application/json",

            "response_schema":
                response_schema,

            "temperature":
                0
        }
    )

    gemini_api_time = (
        time.perf_counter()
        - gemini_api_start_time
    )

    logger.info(
        "Gemini API response received"
    )

    logger.info(
        f"Gemini API time: "
        f"{gemini_api_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # GEMINI RESULT
    # --------------------------------------------------------

    print(
        "\nGemini extraction completed."
    )

    print(
        "\n========== GEMINI RESULT ==========\n"
    )

    print(
        response.text
    )

    # --------------------------------------------------------
    # TOTAL GEMINI TIME
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

def save_result(
    result
):

    start_time = time.perf_counter()

    logger.info("------------------------------------------")
    logger.info("SAVING CASTE CERTIFICATE RESULT")
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
    # CONVERT GEMINI JSON RESPONSE
    # --------------------------------------------------------

    logger.info(
        "Converting Gemini response to JSON..."
    )

    parsed_result = json.loads(
        result
    )

    # --------------------------------------------------------
    # WRITE JSON
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
    # SAVE TIME
    # --------------------------------------------------------

    save_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "CASTE CERTIFICATE RESULT "
        "SAVED SUCCESSFULLY"
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
    logger.info(
        "CASTE CERTIFICATE DATA EXTRACTION STARTED"
    )
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
    # STEP 1
    # ENTERPRISE DOCUMENT OCR
    # ========================================================

    step1_start_time = (
        time.perf_counter()
    )

    document = run_enterprise_ocr(
        INPUT_FILE
    )

    step1_time = (
        time.perf_counter()
        - step1_start_time
    )

    logger.info(
        f"STEP 1 - Enterprise OCR: "
        f"{step1_time:.3f} seconds"
    )

    # ========================================================
    # STEP 2
    # GEMINI 2.5 PRO
    # ========================================================

    step2_start_time = (
        time.perf_counter()
    )

    result = extract_with_gemini(
        document
    )

    step2_time = (
        time.perf_counter()
        - step2_start_time
    )

    logger.info(
        f"STEP 2 - Gemini 2.5 Pro: "
        f"{step2_time:.3f} seconds"
    )

    # ========================================================
    # STEP 3
    # SAVE JSON
    # ========================================================

    step3_start_time = (
        time.perf_counter()
    )

    save_result(
        result
    )

    step3_time = (
        time.perf_counter()
        - step3_start_time
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

    # ========================================================
    # PERFORMANCE SUMMARY
    # ========================================================

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
        "CASTE CERTIFICATE DATA EXTRACTION FINISHED"
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