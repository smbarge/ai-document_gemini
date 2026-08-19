from google.cloud import documentai_v1 as documentai
import json
import os
import time
import logging


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "YOUR_PROJECT_ID"

LOCATION = "us"

PROCESSOR_ID = "YOUR_CUSTOM_EXTRACTOR_PROCESSOR_ID"

PROCESSOR_VERSION = "pretrained-foundation-model-v1.5-2025-05-05"

DOCUMENT_PATH = "../docs/aadhaar/aadhaar_1.jpeg"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# EXTRACT AADHAAR DATA
# ============================================================

def extract_aadhaar_data(document_path):

    # --------------------------------------------------------
    # TOTAL START TIME
    # --------------------------------------------------------

    total_start = time.perf_counter()

    logger.info("==================================================")
    logger.info("STARTING AADHAAR CARD EXTRACTION")
    logger.info("==================================================")

    logger.info(
        "Document path: %s",
        document_path
    )

    # ========================================================
    # CREATE DOCUMENT AI CLIENT
    # ========================================================

    start_time = time.perf_counter()

    logger.info(
        "Initializing Document AI client..."
    )

    client = documentai.DocumentProcessorServiceClient()

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Document AI client initialized in %.3f seconds",
        elapsed
    )

    # ========================================================
    # PROCESSOR
    # ========================================================

    processor_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        "Location: %s",
        LOCATION
    )

    logger.info(
        "Processor version: %s",
        PROCESSOR_VERSION
    )

    # ========================================================
    # READ DOCUMENT
    # ========================================================

    logger.info(
        "Reading Aadhaar document..."
    )

    start_time = time.perf_counter()

    with open(document_path, "rb") as file:
        document_content = file.read()

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Document read completed in %.3f seconds",
        elapsed
    )

    logger.info(
        "Document size: %.2f MB",
        len(document_content) / (1024 * 1024)
    )

    # ========================================================
    # DETECT MIME TYPE
    # ========================================================

    logger.info(
        "Detecting MIME type..."
    )

    start_time = time.perf_counter()

    extension = os.path.splitext(
        document_path
    )[1].lower()

    if extension == ".pdf":

        mime_type = "application/pdf"

    elif extension in [".jpg", ".jpeg"]:

        mime_type = "image/jpeg"

    elif extension == ".png":

        mime_type = "image/png"

    else:

        logger.error(
            "Unsupported file format: %s",
            extension
        )

        raise ValueError(
            "Supported formats: PDF, JPG, JPEG, PNG"
        )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "File extension: %s",
        extension
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    logger.info(
        "MIME detection completed in %.3f seconds",
        elapsed
    )

    # ========================================================
    # CREATE RAW DOCUMENT
    # ========================================================

    logger.info(
        "Creating Document AI request..."
    )

    start_time = time.perf_counter()

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    # ========================================================
    # CREATE REQUEST
    # ========================================================

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Request created in %.3f seconds",
        elapsed
    )

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    logger.info("==================================================")
    logger.info(
        "Sending Aadhaar document to Document AI..."
    )
    logger.info(
        "Custom Extractor processing started"
    )
    logger.info("==================================================")

    start_time = time.perf_counter()

    result = client.process_document(
        request=request
    )

    processing_time = time.perf_counter() - start_time

    logger.info(
        "Document AI processing completed in %.3f seconds",
        processing_time
    )

    document = result.document

    # ========================================================
    # OCR TEXT
    # ========================================================

    logger.info(
        "Reading OCR text..."
    )

    start_time = time.perf_counter()

    ocr_text = document.text

    elapsed = time.perf_counter() - start_time

    logger.info(
        "OCR text read in %.3f seconds",
        elapsed
    )

    logger.info(
        "OCR characters extracted: %d",
        len(ocr_text)
    )

    print(
        "\n================ OCR TEXT ================\n"
    )

    print(ocr_text)

    print(
        "\n===========================================\n"
    )

    # ========================================================
    # FINAL JSON STRUCTURE
    # ========================================================

    logger.info(
        "Creating final JSON structure..."
    )

    start_time = time.perf_counter()

    extracted_data = {
        "name": None,
        "aadhaar_number": None,
        "date_of_birth": None,
        "gender": None,
        "address": None
    }

    elapsed = time.perf_counter() - start_time

    logger.info(
        "JSON structure created in %.3f seconds",
        elapsed
    )

    # ========================================================
    # EXTRACT CUSTOM EXTRACTOR ENTITIES
    # ========================================================

    logger.info(
        "Extracting Custom Extractor entities..."
    )

    start_time = time.perf_counter()

    total_entities = 0
    matched_entities = 0

    for entity in document.entities:

        total_entities += 1

        field_name = entity.type_

        value = entity.mention_text

        confidence = entity.confidence

        logger.info(
            "Entity: %s | Value: %s | Confidence: %.4f",
            field_name,
            value,
            confidence
        )

        if field_name in extracted_data:

            extracted_data[field_name] = value.strip()

            matched_entities += 1

    entity_time = time.perf_counter() - start_time

    logger.info(
        "Entity extraction completed in %.3f seconds",
        entity_time
    )

    logger.info(
        "Total entities detected: %d",
        total_entities
    )

    logger.info(
        "Required entities matched: %d",
        matched_entities
    )

    # ========================================================
    # TOTAL PROCESSING TIME
    # ========================================================

    total_time = time.perf_counter() - total_start

    logger.info("==================================================")
    logger.info(
        "AADHAAR CARD EXTRACTION COMPLETED"
    )
    logger.info("==================================================")

    logger.info(
        "Document AI processing time: %.3f seconds",
        processing_time
    )

    logger.info(
        "Entity extraction time: %.3f seconds",
        entity_time
    )

    logger.info(
        "TOTAL EXECUTION TIME: %.3f seconds",
        total_time
    )

    logger.info("==================================================")

    return extracted_data


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    logger.info(
        "Starting Aadhaar extraction application..."
    )

    data = extract_aadhaar_data(
        DOCUMENT_PATH
    )

    print(
        "\n================ RESULT ================\n"
    )

    print(
        json.dumps(
            data,
            indent=4,
            ensure_ascii=False
        )
    )

    print(
        "\n===========================================\n"
    )