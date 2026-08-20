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

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-pro"
)


# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "docs/panCards/pan1.png"

OUTPUT_FILE = "results/pan1_gemini_result.json"


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

    return mime_types.get(extension)


# ============================================================
# DIRECT GEMINI PAN CARD EXTRACTION
# ============================================================

def extract_with_gemini(file_path):

    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DIRECT GEMINI PAN CARD EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info("Input file: %s", file_path)
    logger.info("Gemini model: %s", GEMINI_MODEL)

    # ========================================================
    # READ IMAGE
    # ========================================================

    logger.info("Reading PAN card image...")

    read_start = time.perf_counter()

    with open(file_path, "rb") as file:
        image_data = file.read()

    read_time = time.perf_counter() - read_start

    logger.info(
        "Image read completed in %.3f seconds",
        read_time
    )

    logger.info(
        "Image size: %.3f MB",
        len(image_data) / (1024 * 1024)
    )

    # ========================================================
    # MIME TYPE
    # ========================================================

    mime_type = get_mime_type(file_path)

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # ========================================================
    # GEMINI CLIENT
    # ========================================================

    logger.info("Initializing Gemini client...")

    client_start = time.perf_counter()

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = time.perf_counter() - client_start

    logger.info(
        "Gemini client initialized in %.3f seconds",
        client_time
    )

    # ========================================================
    # EXTRACTION PROMPT
    # ========================================================

    prompt = """
Analyze the provided PAN card image.

Extract the following information exactly as visible
in the document:

1. name
2. father_name
3. pan_number
4. date_of_birth

Important instructions:

- Read the information directly from the image.
- Extract only information visible in the document.
- Do not invent information.
- Do not guess missing information.
- Do not correct or modify names.
- Preserve values as written in the document.
- If a field is not visible, return an empty string.
- Return only valid JSON.
"""

    logger.info(
        "Gemini extraction prompt prepared."
    )

    # ========================================================
    # CREATE IMAGE PART
    # ========================================================

    logger.info(
        "Preparing PAN card image for Gemini..."
    )

    image_start = time.perf_counter()

    image_part = genai.types.Part.from_bytes(
        data=image_data,
        mime_type=mime_type
    )

    image_time = time.perf_counter() - image_start

    logger.info(
        "Image prepared in %.3f seconds",
        image_time
    )

    # ========================================================
    # GEMINI API
    # ========================================================

    logger.info("-" * 60)
    logger.info("SENDING PAN CARD IMAGE DIRECTLY TO GEMINI")
    logger.info("-" * 60)

    gemini_start = time.perf_counter()

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

                    "name": {
                        "type": "STRING"
                    },

                    "father_name": {
                        "type": "STRING"
                    },

                    "pan_number": {
                        "type": "STRING"
                    },

                    "date_of_birth": {
                        "type": "STRING"
                    }
                },

                "required": [
                    "name",
                    "father_name",
                    "pan_number",
                    "date_of_birth"
                ]
            },

            "temperature": 0
        }
    )

    gemini_time = time.perf_counter() - gemini_start

    logger.info(
        "Gemini API response received in %.3f seconds",
        gemini_time
    )

    # ========================================================
    # GEMINI RESULT
    # ========================================================

    response_text = response.text

    print()
    print("=" * 60)
    print("GEMINI RESULT")
    print("=" * 60)

    print(response_text)

    print("=" * 60)

    logger.info(
        "Gemini response length: %d characters",
        len(response_text)
    )

    # ========================================================
    # JSON CONVERSION
    # ========================================================

    logger.info(
        "Converting Gemini response to JSON..."
    )

    json_start = time.perf_counter()

    extracted_data = json.loads(
        response_text
    )

    json_time = time.perf_counter() - json_start

    logger.info(
        "JSON conversion completed in %.3f seconds",
        json_time
    )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_time = time.perf_counter() - total_start

    logger.info("=" * 60)
    logger.info("PERFORMANCE SUMMARY")
    logger.info("=" * 60)

    logger.info(
        "Image read time : %.3f seconds",
        read_time
    )

    logger.info(
        "Gemini API time : %.3f seconds",
        gemini_time
    )

    logger.info(
        "JSON time       : %.3f seconds",
        json_time
    )

    logger.info(
        "TOTAL TIME      : %.3f seconds",
        total_time
    )

    logger.info("=" * 60)

    return {
        "document": file_path,
        "model": GEMINI_MODEL,
        "gemini_processing_time_seconds": round(
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

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
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
        "Result saved successfully."
    )

    logger.info(
        "Output file: %s",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    application_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("PAN CARD DIRECT GEMINI TEST")
    logger.info("=" * 60)

    logger.info(
        "Input: %s",
        INPUT_FILE
    )

    logger.info(
        "Output: %s",
        OUTPUT_FILE
    )

    # ========================================================
    # INPUT FILE
    # ========================================================

    if not Path(INPUT_FILE).exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    # ========================================================
    # GEMINI EXTRACTION
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
    # APPLICATION TIME
    # ========================================================

    application_time = (
        time.perf_counter()
        - application_start
    )

    logger.info("=" * 60)
    logger.info("APPLICATION COMPLETED")

    logger.info(
        "TOTAL APPLICATION TIME: %.3f seconds",
        application_time
    )

    logger.info("=" * 60)

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