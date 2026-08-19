from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

import json
import os
import time
import logging


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "document-ai-test-506006"

LOCATION = "asia-south1"

PROCESSOR_ID = "ec631cf67f66f191"

PROCESSOR_VERSION = "pretrained-foundation-model-v1.5-2025-08-06"

DOCUMENT_PATH = "docs/casteCertificates/casteCertificate_1.jpg"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# EXTRACT CASTE CERTIFICATE DATA
# ============================================================

def extract_caste_certificate_data(document_path):

    total_start = time.perf_counter()

    logger.info("==================================================")
    logger.info("STARTING CASTE CERTIFICATE EXTRACTION")
    logger.info("==================================================")

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
        "Document path: %s",
        document_path
    )

    # ========================================================
    # CHECK FILE
    # ========================================================

    if not os.path.isfile(document_path):

        raise FileNotFoundError(
            f"Document not found: {document_path}"
        )

    logger.info(
        "Document file found successfully"
    )

    # ========================================================
    # CREATE DOCUMENT AI CLIENT
    # ========================================================

    logger.info(
        "Initializing Document AI client..."
    )

    start_time = time.perf_counter()

    client = documentai.DocumentProcessorServiceClient(
        client_options=ClientOptions(
            api_endpoint=(
                f"{LOCATION}-documentai.googleapis.com"
            )
        )
    )

    logger.info(
        "Document AI client initialized in %.3f seconds",
        time.perf_counter() - start_time
    )

    # ========================================================
    # PROCESSOR NAME
    # ========================================================

    processor_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        "Processor name: %s",
        processor_name
    )

    # ========================================================
    # READ DOCUMENT
    # ========================================================

    logger.info(
        "Reading caste certificate..."
    )

    start_time = time.perf_counter()

    with open(document_path, "rb") as file:

        document_content = file.read()

    logger.info(
        "Document read completed in %.3f seconds",
        time.perf_counter() - start_time
    )

    logger.info(
        "Document size: %.2f MB",
        len(document_content) / (1024 * 1024)
    )

    # ========================================================
    # DETECT MIME TYPE
    # ========================================================

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

        raise ValueError(
            f"Unsupported file format: {extension}. "
            "Supported formats: PDF, JPG, JPEG, PNG"
        )

    logger.info(
        "File extension: %s",
        extension
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # ========================================================
    # CREATE RAW DOCUMENT
    # ========================================================

    logger.info(
        "Creating raw document..."
    )

    start_time = time.perf_counter()

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    logger.info(
        "Raw document created in %.3f seconds",
        time.perf_counter() - start_time
    )

    # ========================================================
    # FIELD DEFINITIONS
    # ========================================================

    field_names = [
        "candidate_name",
        "father_name",
        "mother_name",
        "certificate_number",
        "caste",
        "category",
        "sub_caste",
        "district",
        "state",
        "issue_date",
        "issuing_authority"
    ]

    logger.info(
        "Fields requested: %s",
        ", ".join(field_names)
    )

    # ========================================================
    # CREATE SCHEMA PROPERTIES
    # ========================================================

    properties = []

    for field_name in field_names:

        properties.append(
            documentai.DocumentSchema.EntityType.Property(
                name=field_name,
                value_type="string"
            )
        )

    # ========================================================
    # CREATE SCHEMA OVERRIDE
    # ========================================================

    schema_override = documentai.DocumentSchema(
        display_name="Caste Certificate Schema",
        description="Caste certificate extraction schema",
        entity_types=[
            documentai.DocumentSchema.EntityType(
                name="custom_extraction_document_type",
                base_types=["document"],
                properties=properties
            )
        ]
    )

    logger.info(
        "Schema override created with %d fields",
        len(properties)
    )

    # ========================================================
    # PROCESS OPTIONS
    # ========================================================

    process_options = documentai.ProcessOptions(
        schema_override=schema_override
    )

    # ========================================================
    # CREATE REQUEST
    # ========================================================

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
        process_options=process_options
    )

    logger.info(
        "Document AI request created successfully"
    )

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    logger.info("==================================================")
    logger.info(
        "Sending caste certificate to Document AI..."
    )
    logger.info(
        "Custom Extractor processing started"
    )
    logger.info("==================================================")

    start_time = time.perf_counter()

    result = client.process_document(
        request=request
    )

    processing_time = (
        time.perf_counter() - start_time
    )

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

    logger.info(
        "OCR text read in %.3f seconds",
        time.perf_counter() - start_time
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
    # INITIAL RESULT
    # ========================================================

    extracted_data = {
        "candidate_name": None,
        "father_name": None,
        "mother_name": None,
        "certificate_number": None,
        "caste": None,
        "category": None,
        "sub_caste": None,
        "district": None,
        "state": None,
        "issue_date": None,
        "issuing_authority": None
    }

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

            if value:

                extracted_data[field_name] = (
                    value.strip()
                )

            matched_entities += 1

    entity_time = (
        time.perf_counter() - start_time
    )

    # ========================================================
    # ENTITY STATISTICS
    # ========================================================

    logger.info(
        "Entity extraction completed in %.3f seconds",
        entity_time
    )

    logger.info(
        "Total entities detected: %d",
        total_entities
    )

    logger.info(
        "Required entities matched: %d / %d",
        matched_entities,
        len(extracted_data)
    )

    # ========================================================
    # TOTAL PROCESSING TIME
    # ========================================================

    total_time = (
        time.perf_counter() - total_start
    )

    logger.info("==================================================")
    logger.info(
        "CASTE CERTIFICATE EXTRACTION COMPLETED"
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
        "Starting caste certificate extraction application..."
    )

    try:

        data = extract_caste_certificate_data(
            DOCUMENT_PATH
        )

        print(
            "\n================ FINAL RESULT ================\n"
        )

        print(
            json.dumps(
                data,
                indent=4,
                ensure_ascii=False
            )
        )

        print(
            "\n================================================\n"
        )

    except Exception as error:

        logger.exception(
            "Caste certificate extraction failed: %s",
            error
        )

        raise
