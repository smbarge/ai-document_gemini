from google.cloud import documentai_v1 as documentai
from google.api_core.client_options import ClientOptions

import json
import os
import time
import logging
import traceback


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ID = "document-ai-test-506006"

LOCATION = "asia-south1"

PROCESSOR_ID = "ec631cf67f66f191"

PROCESSOR_VERSION = "pretrained-foundation-model-v1.5-2025-08-06"

DOCUMENT_PATH = "docs/non-creamyLayerCertificates/non-creamyLayer_2.jpg"


# ============================================================
# LOGGING CONFIGURATION
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# EXTRACT NON-CREAMY LAYER DATA
# ============================================================

def extract_ncl_data(document_path):

    total_start = time.perf_counter()

    logger.info("==================================================")
    logger.info(
        "STARTING NON-CREAMY LAYER CERTIFICATE EXTRACTION"
    )
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
    # CHECK DOCUMENT
    # ========================================================

    if not os.path.isfile(document_path):

        logger.error(
            "Document not found: %s",
            document_path
        )

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
            api_endpoint=f"{LOCATION}-documentai.googleapis.com"
        )
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Document AI client initialized in %.3f seconds",
        elapsed
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
        "Reading Non-Creamy Layer certificate..."
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
        "Creating raw document..."
    )

    start_time = time.perf_counter()

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Raw document created in %.3f seconds",
        elapsed
    )

    # ========================================================
    # NCL FIELD DEFINITIONS
    # ========================================================

    field_names = [
        "candidate_name",
        "father_name",
        "mother_name",
        "certificate_number",
        "caste",
        "category",
        "income",
        "financial_year",
        "address",
        "district",
        "state",
        "issue_date",
        "valid_until",
        "issuing_authority"
    ]

    logger.info(
        "NCL fields requested: %s",
        ", ".join(field_names)
    )

    # ========================================================
    # CREATE SCHEMA PROPERTIES
    # ========================================================

    logger.info(
        "Creating NCL schema properties..."
    )

    start_time = time.perf_counter()

    properties = []

    for field_name in field_names:

        properties.append(
            documentai.DocumentSchema.EntityType.Property(
                name=field_name,
                value_type="string"
            )
        )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Schema properties created in %.3f seconds",
        elapsed
    )

    logger.info(
        "Total NCL fields: %d",
        len(properties)
    )

    # ========================================================
    # CREATE SCHEMA OVERRIDE
    # ========================================================

    logger.info(
        "Creating NCL schema override..."
    )

    start_time = time.perf_counter()

    schema_override = documentai.DocumentSchema(
        display_name="Non-Creamy Layer Schema",
        description="Non-Creamy Layer Certificate extraction schema",
        entity_types=[
            documentai.DocumentSchema.EntityType(
                name="custom_extraction_document_type",
                base_types=["document"],
                properties=properties
            )
        ]
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Schema override created in %.3f seconds",
        elapsed
    )

    logger.info(
        "Schema override created with %d fields",
        len(properties)
    )

    # ========================================================
    # PROCESS OPTIONS
    # ========================================================

    logger.info(
        "Creating Document AI process options..."
    )

    start_time = time.perf_counter()

    process_options = documentai.ProcessOptions(
        schema_override=schema_override
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Process options created in %.3f seconds",
        elapsed
    )

    # ========================================================
    # CREATE PROCESS REQUEST
    # ========================================================

    logger.info(
        "Creating Document AI request..."
    )

    start_time = time.perf_counter()

    request = documentai.ProcessRequest(
        name=processor_name,
        raw_document=raw_document,
        process_options=process_options
    )

    elapsed = time.perf_counter() - start_time

    logger.info(
        "Document AI request created successfully"
    )

    logger.info(
        "Request created in %.3f seconds",
        elapsed
    )

    # ========================================================
    # PROCESS DOCUMENT
    # ========================================================

    logger.info("==================================================")

    logger.info(
        "Sending NCL certificate to Document AI..."
    )

    logger.info(
        "Custom Extractor processing started"
    )

    logger.info("==================================================")

    start_time = time.perf_counter()

    try:

        result = client.process_document(
            request=request
        )

    except Exception as error:

        logger.error(
            "NCL extraction failed: %s",
            error
        )

        traceback.print_exc()

        raise

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
        "candidate_name": None,
        "father_name": None,
        "mother_name": None,
        "certificate_number": None,
        "caste": None,
        "category": None,
        "income": None,
        "financial_year": None,
        "address": None,
        "district": None,
        "state": None,
        "issue_date": None,
        "valid_until": None,
        "issuing_authority": None
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

        # ----------------------------------------------------
        # SAVE ONLY VALUE
        # ----------------------------------------------------

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
        "NON-CREAMY LAYER CERTIFICATE EXTRACTION COMPLETED"
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

    try:

        logger.info(
            "Starting NCL certificate extraction application..."
        )

        data = extract_ncl_data(
            DOCUMENT_PATH
        )

        # ====================================================
        # FINAL RESULT
        # ====================================================

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

    except Exception as error:

        logger.error(
            "NCL extraction application failed: %s",
            error
        )

        traceback.print_exc()
