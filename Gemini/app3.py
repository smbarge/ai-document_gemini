import os
import json
import time
import logging
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import HttpOptions


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)

logger = logging.getLogger(__name__)


# ============================================================
# ENVIRONMENT
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

INPUT_FILE = "docs/aadharCards/Adhar1.png"

OUTPUT_FILE = "results/Adhar1_gemini_result.json"


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
# DIRECT GEMINI EXTRACTION
# ============================================================

def extract_with_gemini(file_path):

    total_start = time.perf_counter()

    logger.info("=" * 60)
    logger.info("DIRECT GEMINI AADHAAR EXTRACTION STARTED")
    logger.info("=" * 60)

    logger.info("Input file: %s", file_path)
    logger.info("Gemini model: %s", GEMINI_MODEL)

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    logger.info("Reading Aadhaar image...")

    start_time = time.perf_counter()

    with open(file_path, "rb") as file:
        image_data = file.read()

    read_time = time.perf_counter() - start_time

    logger.info(
        "Image read completed in %.3f seconds",
        read_time
    )

    logger.info(
        "Image size: %.3f MB",
        len(image_data) / (1024 * 1024)
    )

    # --------------------------------------------------------
    # MIME TYPE
    # --------------------------------------------------------

    mime_type = get_mime_type(file_path)

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # --------------------------------------------------------
    # GEMINI CLIENT
    # --------------------------------------------------------

    logger.info("Initializing Gemini client...")

    start_time = time.perf_counter()

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location="global",
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = time.perf_counter() - start_time

    logger.info(
        "Gemini client initialized in %.3f seconds",
        client_time
    )

    # --------------------------------------------------------
    # PROMPT
    # --------------------------------------------------------

    prompt = """
Analyze the provided Aadhaar card image.

Extract the following fields:

1. name
2. aadhaar_number
3. date_of_birth
4. gender
5. address

Instructions:

- Read the information directly from the image.
- Extract only information visible in the document.
- Do not invent information.
- Do not guess missing information.
- Do not correct or modify the extracted values.
- Preserve the text as written in the document.
- If a field is not visible, return an empty string.
- Return only JSON.
"""

    # --------------------------------------------------------
    # CREATE IMAGE PART
    # --------------------------------------------------------

    logger.info(
        "Preparing image for Gemini..."
    )

    image_part = genai.types.Part.from_bytes(
        data=image_data,
        mime_type=mime_type
    )

    # --------------------------------------------------------
    # GEMINI API
    # --------------------------------------------------------

    logger.info("-" * 60)
    logger.info("SENDING IMAGE DIRECTLY TO GEMINI")
    logger.info("-" * 60)

    start_time = time.perf_counter()

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
                    "aadhaar_number": {
                        "type": "STRING"
                    },
                    "date_of_birth": {
                        "type": "STRING"
                    },
                    "gender": {
                        "type": "STRING"
                    },
                    "address": {
                        "type": "STRING"
                    }
                },
                "required": [
                    "name",
                    "aadhaar_number",
                    "date_of_birth",
                    "gender",
                    "address"
                ]
            },
            "temperature": 0
        }
    )

    gemini_time = time.perf_counter() - start_time

    logger.info(
        "Gemini response received in %.3f seconds",
        gemini_time
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    response_text = response.text

    print()
    print("=" * 60)
    print("GEMINI RESULT")
    print("=" * 60)
    print(response_text)
    print("=" * 60)

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    extracted_data = json.loads(
        response_text
    )

    # --------------------------------------------------------
    # TOTAL TIME
    # --------------------------------------------------------

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
        "Total time      : %.3f seconds",
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
        "Result saved: %s",
        output_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 60)
    logger.info("AADHAAR DIRECT GEMINI TEST")
    logger.info("=" * 60)

    if not Path(INPUT_FILE).exists():

        raise FileNotFoundError(
            f"File not found: {INPUT_FILE}"
        )

    result = extract_with_gemini(
        INPUT_FILE
    )

    save_result(
        result
    )

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
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        logger.exception(
            "APPLICATION FAILED: %s",
            error
        )