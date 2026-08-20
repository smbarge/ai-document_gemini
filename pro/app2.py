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

DOCUMENT_PATH = "../docs/12thMarkSheets/HSC_1.jpeg"

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

    extension = os.path.splitext(
        document_path
    )[1].lower()

    mime_types = {

        ".pdf": "application/pdf",

        ".jpg": "image/jpeg",

        ".jpeg": "image/jpeg",

        ".png": "image/png",

        ".tif": "image/tiff",

        ".tiff": "image/tiff"
    }

    return mime_types.get(
        extension,
        "application/octet-stream"
    )


# ============================================================
# CREATE DOCUMENT AI CLIENT
# ============================================================

def create_document_ai_client():

    logger.info(
        "Initializing Document AI client..."
    )

    start_time = time.perf_counter()

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

    logger.info(
        "Document AI client initialized in %.3f seconds",
        time.perf_counter() - start_time
    )

    return client


# ============================================================
# CREATE PROCESSOR VERSION RESOURCE NAME
# ============================================================

def create_processor_name():

    processor_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    return processor_name


# ============================================================
# CREATE HSC EXTRACTION SCHEMA
# ============================================================

def create_schema_override():

    field_names = [

        "candidate_name",

        "seat_number",

        "mother_name",

        "percentage",

        "passing_year",

        "total_marks",

        "obtained_marks",

        "stream"
    ]

    logger.info(
        "Fields requested: %s",
        ", ".join(field_names)
    )

    properties = []

    for field_name in field_names:

        properties.append(

            documentai.DocumentSchema.EntityType.Property(

                name=field_name,

                value_type="string"
            )
        )

    schema_override = documentai.DocumentSchema(

        display_name="HSC Schema",

        description=(
            "12th HSC Marksheet extraction schema"
        ),

        entity_types=[

            documentai.DocumentSchema.EntityType(

                name="custom_extraction_document_type",

                base_types=["document"],

                properties=properties
            )
        ]
    )

    logger.info(
        "HSC schema created with %d fields",
        len(properties)
    )

    return schema_override


# ============================================================
# EXTRACT 12TH MARKSHEET DATA
# ============================================================

def extract_12th_marksheet_data(document_path):

    total_start = time.perf_counter()

    logger.info("")
    logger.info(
        "=================================================="
    )

    logger.info(
        "STARTING 12TH MARKSHEET EXTRACTION"
    )

    logger.info(
        "=================================================="
    )

    # ========================================================
    # CONFIGURATION LOGS
    # ========================================================

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

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    # ========================================================
    # READ DOCUMENT
    # ========================================================

    logger.info(
        "Reading 12th marksheet..."
    )

    start_time = time.perf_counter()

    with open(
        document_path,
        "rb"
    ) as file:

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
    # MIME TYPE
    # ========================================================

    mime_type = get_mime_type(
        document_path
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # ========================================================
    # CREATE DOCUMENT AI CLIENT
    # ========================================================

    client = create_document_ai_client()

    # ========================================================
    # PROCESSOR NAME
    # ========================================================

    processor_name = create_processor_name()

    logger.info(
        "Processor name: %s",
        processor_name
    )

    # ========================================================
    # RAW DOCUMENT
    # ========================================================

    raw_document = documentai.RawDocument(

        content=document_content,

        mime_type=mime_type
    )

    logger.info(
        "Raw document created successfully"
    )

    # ========================================================
    # SCHEMA
    # ========================================================

    schema_override = create_schema_override()

    # ========================================================
    # PROCESS OPTIONS
    # ========================================================

    process_options = documentai.ProcessOptions(

        schema_override=schema_override
    )

    # ========================================================
    # REQUEST
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

    logger.info("")
    logger.info(
        "=================================================="
    )

    logger.info(
        "SENDING 12TH MARKSHEET TO DOCUMENT AI"
    )

    logger.info(
        "CUSTOM EXTRACTOR PROCESSING STARTED"
    )

    logger.info(
        "=================================================="
    )

    processing_start = time.perf_counter()

    result = client.process_document(
        request=request
    )

    processing_time = (
        time.perf_counter()
        - processing_start
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

    ocr_text = document.text

    logger.info(
        "OCR characters extracted: %d",
        len(ocr_text)
    )

    print(
        "\n================ OCR TEXT ================\n"
    )

    print(
        ocr_text
    )

    print(
        "\n===========================================\n"
    )

    # ========================================================
    # INITIAL RESULT
    # ========================================================

    extracted_data = {

        "candidate_name": None,

        "seat_number": None,

        "mother_name": None,

        "percentage": None,

        "passing_year": None,

        "total_marks": None,

        "obtained_marks": None,

        "stream": None
    }

    # ========================================================
    # DOCUMENT AI ENTITIES
    # ========================================================

    logger.info(
        "Extracting Custom Extractor entities..."
    )

    entity_start = time.perf_counter()

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

    entity_time = (
        time.perf_counter()
        - entity_start
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
    # DOCUMENT AI RESULT
    # ========================================================

    print(
        "\n================ DOCUMENT AI RESULT ================\n"
    )

    print(
        json.dumps(
            extracted_data,
            indent=4,
            ensure_ascii=False
        )
    )

    print(
        "\n=====================================================\n"
    )

    total_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        "Document AI processing time: %.3f seconds",
        processing_time
    )

    logger.info(
        "Entity extraction time: %.3f seconds",
        entity_time
    )

    logger.info(
        "TOTAL DOCUMENT AI TIME: %.3f seconds",
        total_time
    )

    return extracted_data, ocr_text


# ============================================================
# GEMINI 2.5 PRO EXTRACTION
# ============================================================

def extract_with_gemini(
    project_id,
    ocr_text,
    document_ai_result
):

    logger.info("")
    logger.info(
        "=================================================="
    )

    logger.info(
        "STARTING GEMINI 2.5 PRO EXTRACTION"
    )

    logger.info(
        "=================================================="
    )

    start_time = time.perf_counter()

    # ========================================================
    # RESPONSE SCHEMA
    # ========================================================

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

            "percentage": {
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

            "stream": {
                "type": "STRING"
            }
        },

        "required": [

            "candidate_name",

            "seat_number",

            "mother_name",

            "percentage",

            "passing_year",

            "total_marks",

            "obtained_marks",

            "stream"
        ]
    }

    # ========================================================
    # PROMPT
    # ========================================================

    prompt = f"""
You are a document data extraction system.

You are processing a 12th standard HSC marksheet.

Document AI has already processed the document.

Extract the following fields:

1. candidate_name
2. seat_number
3. mother_name
4. percentage
5. passing_year
6. total_marks
7. obtained_marks
8. stream

Rules:

- Use only information present in the OCR text or
  Document AI extracted data.
- Do not invent values.
- Do not guess missing information.
- Do not change the spelling of names.
- Preserve values exactly as written.
- If a value cannot be identified, return an empty string.
- Return only valid JSON.

DOCUMENT AI EXTRACTED DATA:

{json.dumps(
    document_ai_result,
    indent=4,
    ensure_ascii=False
)}

OCR TEXT:

------------------------------
{ocr_text}
------------------------------
"""

    # ========================================================
    # CREATE GEMINI CLIENT
    # ========================================================

    logger.info(
        "Initializing Gemini Vertex AI client..."
    )

    client = genai.Client(

        vertexai=True,

        project=project_id,

        location="global",

        http_options=HttpOptions(
            api_version="v1"
        )
    )

    logger.info(
        "Gemini client initialized successfully"
    )

    # ========================================================
    # GEMINI REQUEST
    # ========================================================

    logger.info(
        "Sending data to Gemini 2.5 Pro..."
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

    gemini_time = (
        time.perf_counter()
        - gemini_start
    )

    logger.info(
        "Gemini response received in %.3f seconds",
        gemini_time
    )

    # ========================================================
    # PARSE JSON
    # ========================================================

    result = json.loads(
        response.text
    )

    # ========================================================
    # PRINT GEMINI RESULT
    # ========================================================

    print(
        "\n================ GEMINI RESULT ================\n"
    )

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print(
        "\n================================================\n"
    )

    logger.info(
        "Gemini extraction completed in %.3f seconds",
        time.perf_counter() - start_time
    )

    return result


# ============================================================
# SAVE JSON
# ============================================================

def save_json(result):

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
    logger.info(
        "============================================================"
    )

    logger.info(
        "12TH MARKSHEET DATA EXTRACTION STARTED"
    )

    logger.info(
        "============================================================"
    )

    try:

        # ====================================================
        # DISPLAY CONFIGURATION
        # ====================================================

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
            DOCUMENT_PATH
        )

        logger.info(
            "Output        : %s",
            OUTPUT_FILE
        )

        # ====================================================
        # STEP 1 - DOCUMENT AI
        # ====================================================

        logger.info("")
        logger.info(
            "============================================================"
        )

        logger.info(
            "STEP 1 - DOCUMENT AI CUSTOM EXTRACTOR"
        )

        logger.info(
            "============================================================"
        )

        document_ai_result, ocr_text = (
            extract_12th_marksheet_data(
                DOCUMENT_PATH
            )
        )

        # ====================================================
        # STEP 2 - GEMINI
        # ====================================================

        logger.info("")
        logger.info(
            "============================================================"
        )

        logger.info(
            "STEP 2 - GEMINI 2.5 PRO"
        )

        logger.info(
            "============================================================"
        )

        final_result = extract_with_gemini(

            PROJECT_ID,

            ocr_text,

            document_ai_result
        )

        # ====================================================
        # STEP 3 - SAVE JSON
        # ====================================================

        logger.info("")
        logger.info(
            "============================================================"
        )

        logger.info(
            "STEP 3 - SAVE JSON"
        )

        logger.info(
            "============================================================"
        )

        save_json(
            final_result
        )

        # ====================================================
        # STEP 4 - FINAL OUTPUT
        # ====================================================

        print(
            "\n"
            + "=" * 60
        )

        print(
            "FINAL JSON OUTPUT"
        )

        print(
            "=" * 60
        )

        print(
            json.dumps(
                final_result,
                indent=4,
                ensure_ascii=False
            )
        )

        print(
            "=" * 60
        )

        print(
            "JSON FILE:"
        )

        print(
            os.path.abspath(
                OUTPUT_FILE
            )
        )

        print(
            "=" * 60
        )

        # ====================================================
        # PERFORMANCE
        # ====================================================

        total_time = (
            time.perf_counter()
            - application_start
        )

        logger.info("")
        logger.info(
            "============================================================"
        )

        logger.info(
            "PERFORMANCE SUMMARY"
        )

        logger.info(
            "============================================================"
        )

        logger.info(
            "TOTAL EXECUTION TIME: %.3f seconds",
            total_time
        )

        logger.info(
            "============================================================"
        )

        logger.info(
            "12TH MARKSHEET DATA EXTRACTION FINISHED"
        )

        logger.info(
            "============================================================"
        )

    except Exception as error:

        logger.exception(
            "12th marksheet extraction failed: %s",
            error
        )

        raise


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()