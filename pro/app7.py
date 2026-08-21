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

INPUT_FILE = "docs/domicileCertificates/domicile_2.jpg"

OUTPUT_FILE = "output/domicile_certificate_result.json"


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
# MIME TYPE
# ============================================================

def get_mime_type(file_path):

    extension = os.path.splitext(file_path)[1].lower()

    if extension == ".jpg" or extension == ".jpeg":
        return "image/jpeg"

    if extension == ".png":
        return "image/png"

    if extension == ".pdf":
        return "application/pdf"

    if extension == ".tif" or extension == ".tiff":
        return "image/tiff"

    return "application/octet-stream"


# ============================================================
# ENTERPRISE DOCUMENT AI OCR
# ============================================================

def run_enterprise_ocr(file_path):

    logger.info("")
    logger.info("=" * 60)
    logger.info("ENTERPRISE DOCUMENT AI OCR STARTED")
    logger.info("DOMICILE CERTIFICATE")
    logger.info("=" * 60)

    logger.info(
        "Input file: %s",
        file_path
    )

    mime_type = get_mime_type(file_path)

    logger.info(
        "MIME type: %s",
        mime_type
    )

    client_options = ClientOptions(
        api_endpoint=f"{LOCATION}-documentai.googleapis.com"
    )

    client = documentai.DocumentProcessorServiceClient(
        client_options=client_options
    )

    processor_version_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        "Processor: %s",
        processor_version_name
    )

    with open(file_path, "rb") as file:

        document_content = file.read()

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    request = documentai.ProcessRequest(
        name=processor_version_name,
        raw_document=raw_document
    )

    logger.info(
        "Sending domicile certificate to Enterprise Document AI..."
    )

    start_time = time.perf_counter()

    response = client.process_document(
        request=request
    )

    processing_time = (
        time.perf_counter()
        - start_time
    )

    ocr_text = response.document.text

    logger.info(
        "Enterprise OCR completed in %.3f seconds",
        processing_time
    )

    logger.info(
        "OCR text length: %d characters",
        len(ocr_text)
    )

    print("\n")
    print("=" * 60)
    print("OCR TEXT")
    print("=" * 60)
    print(ocr_text)
    print("=" * 60)

    logger.info(
        "ENTERPRISE DOCUMENT AI OCR FINISHED"
    )

    return ocr_text


# ============================================================
# GEMINI 2.5 PRO
# ============================================================

def extract_with_gemini(ocr_text):

    logger.info("")
    logger.info("=" * 60)
    logger.info("GEMINI 2.5 PRO EXTRACTION STARTED")
    logger.info("DOMICILE CERTIFICATE")
    logger.info("=" * 60)

    prompt = f"""
You are an expert document data extraction system.

The following OCR text belongs to an Indian domicile certificate.

Extract the following fields:

1. candidate_name
2. certificate_number
3. issue_date
4. date_of_birth
5. father_name
6. mother_name
7. address
8. village
9. taluka
10. district
11. state
12. domicile_statement
13. issuing_authority

Instructions:

- Use only information present in the OCR text.
- Do not invent information.
- Do not guess missing information.
- Do not calculate anything.
- Do not infer missing information.
- Do not correct spelling.
- Preserve names exactly as written.
- Preserve certificate number exactly as written.
- Preserve dates exactly as written.
- Preserve address exactly as written.
- Preserve village, taluka, district and state exactly as written.
- Preserve the domicile statement exactly as written.
- Preserve the issuing authority exactly as written.
- If a field is not present, return an empty string.
- Return only valid JSON.
- Do not return markdown.
- Do not return explanations.
- Do not add extra fields.

Return JSON in exactly this structure:

{{
    "candidate_name": "",
    "certificate_number": "",
    "issue_date": "",
    "date_of_birth": "",
    "father_name": "",
    "mother_name": "",
    "address": "",
    "village": "",
    "taluka": "",
    "district": "",
    "state": "",
    "domicile_statement": "",
    "issuing_authority": ""
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

    print("\n")
    print("=" * 60)
    print("GEMINI RESULT")
    print("=" * 60)
    print(response.text)
    print("=" * 60)

    logger.info(
        "GEMINI 2.5 PRO EXTRACTION FINISHED"
    )

    return response.text


# ============================================================
# SAVE JSON
# ============================================================

def save_result(result):

    logger.info("")
    logger.info("=" * 60)
    logger.info("SAVING DOMICILE CERTIFICATE RESULT")
    logger.info("=" * 60)

    output_directory = os.path.dirname(
        OUTPUT_FILE
    )

    if output_directory:

        os.makedirs(
            output_directory,
            exist_ok=True
        )

    parsed_result = json.loads(
        result
    )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            parsed_result,
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
    logger.info("DOMICILE CERTIFICATE DATA EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info(
        "Project       : %s",
        PROJECT_ID
    )

    logger.info(
        "Location      : %s",
        LOCATION
    )

    logger.info(
        "Processor ID  : %s",
        PROCESSOR_ID
    )

    logger.info(
        "Processor Ver : %s",
        PROCESSOR_VERSION
    )

    logger.info(
        "Gemini Model  : %s",
        GEMINI_MODEL
    )

    logger.info(
        "Input         : %s",
        INPUT_FILE
    )

    logger.info(
        "Output        : %s",
        OUTPUT_FILE
    )

    # ========================================================
    # STEP 1 - ENTERPRISE DOCUMENT AI
    # ========================================================

    ocr_text = run_enterprise_ocr(
        INPUT_FILE
    )

    # ========================================================
    # STEP 2 - GEMINI 2.5 PRO
    # ========================================================

    result = extract_with_gemini(
        ocr_text
    )

    # ========================================================
    # STEP 3 - SAVE JSON
    # ========================================================

    save_result(
        result
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
            json.loads(result),
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 60)

    total_time = (
        time.perf_counter()
        - application_start
    )

    logger.info(
        "TOTAL EXECUTION TIME: %.3f seconds",
        total_time
    )

    logger.info("=" * 60)
    logger.info(
        "DOMICILE CERTIFICATE DATA EXTRACTION FINISHED"
    )
    logger.info("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()