import os
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
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


# ============================================================
# GOOGLE CLOUD / DOCUMENT AI CONFIGURATION
# ============================================================

PROJECT_ID = "document-ai-test-506006"

LOCATION = "asia-south1"

PROCESSOR_ID = "ec631cf67f66f191"

PROCESSOR_VERSION = "pretrained-foundation-model-v1.5-2025-08-06"

GEMINI_MODEL = "gemini-2.5-pro"


# ============================================================
# FILE CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(
    "/home/user/Desktop/karishma/ai-document_gemini/Gemini"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "DOCS"
    / "10thMarkSheets"
    / "SSC_1.jpeg"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "ssc_1_gemini_result.json"
)


# ============================================================
# MIME TYPE
# ============================================================

def get_mime_type(file_path):

    extension = Path(file_path).suffix.lower()

    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".pdf": "application/pdf",
        ".tif": "image/tiff",
        ".tiff": "image/tiff"
    }

    mime_type = mime_types.get(extension)

    if not mime_type:

        raise ValueError(
            f"Unsupported file type: {extension}"
        )

    return mime_type


# ============================================================
# DIRECT GEMINI DOCUMENT EXTRACTION
# ============================================================

def extract_with_gemini(file_path):

    total_start_time = time.perf_counter()

    logger.info("=" * 60)
    logger.info(
        "DIRECT GEMINI DOCUMENT EXTRACTION STARTED"
    )
    logger.info("=" * 60)

    logger.info(
        "Input document: %s",
        file_path
    )

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
        "Gemini model: %s",
        GEMINI_MODEL
    )


    # ========================================================
    # READ DOCUMENT
    # ========================================================

    logger.info(
        "Reading document..."
    )

    read_start_time = time.perf_counter()

    with open(
        file_path,
        "rb"
    ) as file:

        document_content = file.read()

    read_time = (
        time.perf_counter()
        - read_start_time
    )

    logger.info(
        "Document read completed in %.3f seconds",
        read_time
    )

    logger.info(
        "Document size: %.3f MB",
        len(document_content) / (1024 * 1024)
    )


    # ========================================================
    # MIME TYPE
    # ========================================================

    mime_type = get_mime_type(
        file_path
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )


    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    logger.info(
        "Initializing Gemini client..."
    )

    client_start_time = time.perf_counter()

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = (
        time.perf_counter()
        - client_start_time
    )

    logger.info(
        "Gemini client initialized in %.3f seconds",
        client_time
    )


    # ========================================================
    # EXTRACTION PROMPT
    # ========================================================

    prompt = """
Analyze the provided 10th standard marksheet image.

Extract the following information exactly as visible
in the document:

1. candidate_name
2. seat_number
3. mother_name
4. percentage
5. passing_year

Important instructions:

- Read the information directly from the document image.
- Extract only information that is visible.
- Do not invent information.
- Do not guess missing information.
- Do not calculate the percentage.
- Do not modify names.
- Preserve the values as written in the document.
- If a field is not visible, return an empty string.
- Return ONLY valid JSON.
"""

    logger.info(
        "Gemini extraction prompt prepared"
    )


    # ========================================================
    # IMAGE INPUT
    # ========================================================

    logger.info(
        "Preparing document image for Gemini..."
    )

    image_start_time = time.perf_counter()

    image_part = genai.types.Part.from_bytes(
        data=document_content,
        mime_type=mime_type
    )

    image_time = (
        time.perf_counter()
        - image_start_time
    )

    logger.info(
        "Image prepared in %.3f seconds",
        image_time
    )


    # ========================================================
    # DIRECT GEMINI API
    # ========================================================

    logger.info("-" * 60)
    logger.info(
        "SENDING IMAGE DIRECTLY TO GEMINI"
    )
    logger.info("-" * 60)

    gemini_start_time = time.perf_counter()

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            image_part,
            prompt
        ],
        config={
            "response_mime_type": "application/json",

            "response_schema": {
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

                    "percentage": {
                        "type": "STRING"
                    },

                    "passing_year": {
                        "type": "STRING"
                    }
                },

                "required": [
                    "candidate_name",
                    "seat_number",
                    "mother_name",
                    "percentage",
                    "passing_year"
                ]
            },

            "temperature": 0
        }
    )

    gemini_time = (
        time.perf_counter()
        - gemini_start_time
    )

    logger.info(
        "Gemini API response received in %.3f seconds",
        gemini_time
    )


    # ========================================================
    # GEMINI RAW RESPONSE
    # ========================================================

    response_text = response.text

    logger.info(
        "Gemini response received"
    )

    logger.info(
        "Response length: %d characters",
        len(response_text)
    )

    print()
    print("=" * 60)
    print("GEMINI RAW RESULT")
    print("=" * 60)
    print(response_text)
    print("=" * 60)


    # ========================================================
    # JSON CONVERSION
    # ========================================================

    logger.info(
        "Converting Gemini response to JSON..."
    )

    json_start_time = time.perf_counter()

    extracted_data = json.loads(
        response_text
    )

    json_time = (
        time.perf_counter()
        - json_start_time
    )

    logger.info(
        "JSON conversion completed in %.3f seconds",
        json_time
    )


    # ========================================================
    # TOTAL EXECUTION TIME
    # ========================================================

    total_time = (
        time.perf_counter()
        - total_start_time
    )

    logger.info("=" * 60)
    logger.info(
        "PERFORMANCE SUMMARY"
    )
    logger.info("=" * 60)

    logger.info(
        "Document read time : %.3f seconds",
        read_time
    )

    logger.info(
        "Client init time   : %.3f seconds",
        client_time
    )

    logger.info(
        "Image preparation  : %.3f seconds",
        image_time
    )

    logger.info(
        "Gemini API time    : %.3f seconds",
        gemini_time
    )

    logger.info(
        "JSON conversion    : %.3f seconds",
        json_time
    )

    logger.info(
        "TOTAL TIME         : %.3f seconds",
        total_time
    )

    logger.info("=" * 60)


    # ========================================================
    # RESULT
    # ========================================================

    return {
        "project_id": PROJECT_ID,
        "location": LOCATION,
        "processor_id": PROCESSOR_ID,
        "processor_version": PROCESSOR_VERSION,
        "document": str(file_path),
        "model": GEMINI_MODEL,
        "processing_time_seconds": round(
            gemini_time,
            3
        ),
        "total_execution_time_seconds": round(
            total_time,
            3
        ),
        "extracted_data": extracted_data
    }


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    logger.info(
        "Saving extraction result..."
    )

    save_start_time = time.perf_counter()

    OUTPUT_FILE.parent.mkdir(
        parents=True,
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
        - save_start_time
    )

    logger.info(
        "Result saved successfully"
    )

    logger.info(
        "Output file: %s",
        OUTPUT_FILE
    )

    logger.info(
        "JSON save time: %.3f seconds",
        save_time
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application_start_time = (
        time.perf_counter()
    )

    logger.info("")
    logger.info("=" * 60)
    logger.info(
        "10TH MARKSHEET DIRECT GEMINI TEST"
    )
    logger.info("=" * 60)

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
        "Gemini Model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "Project root: %s",
        PROJECT_ROOT
    )

    logger.info(
        "Input: %s",
        INPUT_FILE
    )

    logger.info(
        "Output: %s",
        OUTPUT_FILE
    )


    # ========================================================
    # CHECK INPUT FILE
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input document not found: {INPUT_FILE}"
        )

    logger.info(
        "Input document exists"
    )


    # ========================================================
    # EXTRACT
    # ========================================================

    result = extract_with_gemini(
        INPUT_FILE
    )


    # ========================================================
    # SAVE
    # ========================================================

    save_result(
        result
    )


    # ========================================================
    # APPLICATION TOTAL TIME
    # ========================================================

    application_time = (
        time.perf_counter()
        - application_start_time
    )

    logger.info("=" * 60)
    logger.info(
        "APPLICATION COMPLETED"
    )

    logger.info(
        "TOTAL APPLICATION TIME: %.3f seconds",
        application_time
    )

    logger.info("=" * 60)


    # ========================================================
    # FINAL EXTRACTED DATA
    # ========================================================

    print()
    print("=" * 60)
    print("FINAL EXTRACTED DATA")
    print("=" * 60)

    print(
        json.dumps(
            result["extracted_data"],
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 60)


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logger.exception(
            "APPLICATION FAILED: %s",
            error
        )