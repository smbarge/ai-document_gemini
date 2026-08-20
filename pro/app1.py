import json
import time
import logging
from pathlib import Path

from google.api_core.client_options import ClientOptions
from google.cloud import documentai_v1 as documentai
from google.cloud import resourcemanager_v3

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
# GOOGLE CLOUD CONFIGURATION
# ============================================================

LOCATION = "us"

PROCESSOR_VERSION = "stable"

GEMINI_MODEL = "gemini-2.5-pro"


# ============================================================
# FILE CONFIGURATION
# ============================================================

INPUT_FILE = "docs/10thMarkSheets/SSC_1.jpeg"

OUTPUT_FILE = "output/marksheet_result.json"


# ============================================================
# GET GOOGLE CLOUD PROJECT AUTOMATICALLY
# ============================================================

def get_project_id():

    logger.info("------------------------------------------")
    logger.info("GETTING GOOGLE CLOUD PROJECT")
    logger.info("------------------------------------------")

    try:

        from google.auth import default

        credentials, project_id = default()

        if not project_id:

            raise RuntimeError(
                "Google Cloud project could not be detected "
                "from Application Default Credentials."
            )

        logger.info(
            f"Google Cloud Project detected: {project_id}"
        )

        return project_id

    except Exception as error:

        logger.exception(
            "Unable to detect Google Cloud project."
        )

        raise RuntimeError(
            "Could not determine Google Cloud project. "
            "Make sure Google Cloud authentication is configured."
        ) from error


# ============================================================
# GET DOCUMENT AI PROCESSOR AUTOMATICALLY
# ============================================================

def get_processor_id(project_id):

    logger.info("------------------------------------------")
    logger.info("SEARCHING FOR DOCUMENT AI PROCESSOR")
    logger.info("------------------------------------------")

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

    parent = (
        f"projects/{project_id}"
        f"/locations/{LOCATION}"
    )

    logger.info(
        f"Searching processors under: {parent}"
    )

    processors = list(
        client.list_processors(
            parent=parent
        )
    )

    if not processors:

        raise RuntimeError(
            f"No Document AI processors found in "
            f"project '{project_id}' "
            f"and location '{LOCATION}'."
        )

    logger.info(
        f"Found {len(processors)} Document AI processor(s)."
    )

    # --------------------------------------------------------
    # If multiple processors exist, display them.
    # --------------------------------------------------------

    for index, processor in enumerate(
        processors,
        start=1
    ):

        logger.info(
            f"Processor {index}: "
            f"name={processor.name}, "
            f"display_name={processor.display_name}, "
            f"type={processor.type_}"
        )

    # --------------------------------------------------------
    # Use the first processor automatically.
    # --------------------------------------------------------

    selected_processor = processors[0]

    processor_id = (
        selected_processor.name.split("/")[-1]
    )

    logger.info(
        "Selected Document AI processor:"
    )

    logger.info(
        f"Processor ID: {processor_id}"
    )

    logger.info(
        f"Processor Name: "
        f"{selected_processor.display_name}"
    )

    return processor_id


# ============================================================
# MIME TYPE
# ============================================================

def get_mime_type(file_path):

    extension = (
        Path(file_path)
        .suffix
        .lower()
    )

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
# INPUT FILE VALIDATION
# ============================================================

def validate_input_file():

    logger.info("------------------------------------------")
    logger.info("VALIDATING INPUT FILE")
    logger.info("------------------------------------------")

    input_path = Path(
        INPUT_FILE
    )

    logger.info(
        f"Input path: {input_path}"
    )

    logger.info(
        f"Absolute path: "
        f"{input_path.resolve()}"
    )

    if not input_path.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{input_path.resolve()}"
        )

    if not input_path.is_file():

        raise ValueError(
            f"Input path is not a file: "
            f"{input_path.resolve()}"
        )

    file_size_mb = (

        input_path.stat().st_size

        / (1024 * 1024)
    )

    logger.info(
        "Input file exists."
    )

    logger.info(
        f"Input file size: "
        f"{file_size_mb:.3f} MB"
    )


# ============================================================
# DOCUMENT AI OCR
# ============================================================

def run_enterprise_ocr(
    project_id,
    processor_id,
    file_path
):

    start_time = (
        time.perf_counter()
    )

    logger.info("------------------------------------------")
    logger.info(
        "ENTERPRISE DOCUMENT OCR STARTED"
    )
    logger.info("------------------------------------------")

    logger.info(
        f"Input file: {file_path}"
    )

    # --------------------------------------------------------
    # File
    # --------------------------------------------------------

    file_path_object = Path(
        file_path
    )

    if not file_path_object.exists():

        raise FileNotFoundError(
            f"Input file not found: "
            f"{file_path_object.resolve()}"
        )

    file_size_mb = (

        file_path_object.stat().st_size

        / (1024 * 1024)
    )

    logger.info(
        f"File size: "
        f"{file_size_mb:.3f} MB"
    )

    # --------------------------------------------------------
    # MIME
    # --------------------------------------------------------

    mime_type = get_mime_type(
        file_path
    )

    if not mime_type:

        raise ValueError(
            f"Unsupported file type: "
            f"{file_path}"
        )

    logger.info(
        f"MIME type: {mime_type}"
    )

    # --------------------------------------------------------
    # Document AI client
    # --------------------------------------------------------

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
        "Document AI client initialized"
    )

    # --------------------------------------------------------
    # Processor resource
    # --------------------------------------------------------

    processor_name = (

        f"projects/{project_id}"

        f"/locations/{LOCATION}"

        f"/processors/{processor_id}"
    )

    logger.info(
        f"Processor resource: "
        f"{processor_name}"
    )

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    logger.info(
        "Reading input document..."
    )

    with open(
        file_path,
        "rb"
    ) as file:

        document_content = (
            file.read()
        )

    logger.info(
        "Input document read successfully"
    )

    # --------------------------------------------------------
    # Raw document
    # --------------------------------------------------------

    raw_document = (
        documentai.RawDocument(

            content=document_content,

            mime_type=mime_type
        )
    )

    # --------------------------------------------------------
    # OCR configuration
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
    # Request
    # --------------------------------------------------------

    request = documentai.ProcessRequest(

        name=processor_name,

        raw_document=raw_document,

        process_options=process_options
    )

    # --------------------------------------------------------
    # Process document
    # --------------------------------------------------------

    logger.info(
        "Sending document to "
        "Enterprise Document OCR..."
    )

    ocr_api_start = (
        time.perf_counter()
    )

    response = (
        client.process_document(
            request=request
        )
    )

    ocr_api_time = (

        time.perf_counter()

        - ocr_api_start
    )

    logger.info(
        "Enterprise OCR API response "
        f"received in "
        f"{ocr_api_time:.3f} seconds"
    )

    document = (
        response.document
    )

    # --------------------------------------------------------
    # OCR information
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
        f"Pages detected: "
        f"{page_count}"
    )

    logger.info(
        f"OCR text length: "
        f"{ocr_text_length} characters"
    )

    # --------------------------------------------------------
    # Print OCR
    # --------------------------------------------------------

    print(
        "\n========== OCR TEXT ==========\n"
    )

    print(
        document.text
    )

    print(
        "\n========== END OCR TEXT ==========\n"
    )

    # --------------------------------------------------------
    # Total OCR time
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
    logger.info(
        "ENTERPRISE DOCUMENT OCR FINISHED"
    )
    logger.info("------------------------------------------")

    return document


# ============================================================
# GEMINI EXTRACTION
# ============================================================

def extract_with_gemini(
    project_id,
    ocr_document
):

    start_time = (
        time.perf_counter()
    )

    logger.info("------------------------------------------")
    logger.info(
        "GEMINI 2.5 PRO EXTRACTION STARTED"
    )
    logger.info("------------------------------------------")

    logger.info(
        f"Gemini model: {GEMINI_MODEL}"
    )

    # --------------------------------------------------------
    # OCR text
    # --------------------------------------------------------

    ocr_text = (
        ocr_document.text
    )

    logger.info(
        f"OCR text sent to Gemini: "
        f"{len(ocr_text)} characters"
    )

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    response_schema = {

        "type": "OBJECT",

        "properties": {

            "candidate_name": {

                "type": "STRING",

                "description": (
                    "Full name of the "
                    "student/candidate "
                    "as written on the "
                    "10th marksheet."
                )
            },

            "seat_number": {

                "type": "STRING",

                "description": (
                    "Seat number, examination "
                    "number, or roll number."
                )
            },

            "mother_name": {

                "type": "STRING",

                "description": (
                    "Mother's name exactly "
                    "as written on the marksheet."
                )
            },

            "percentage": {

                "type": "STRING",

                "description": (
                    "Overall percentage "
                    "obtained by the student."
                )
            },

            "passing_year": {

                "type": "STRING",

                "description": (
                    "Year in which the "
                    "student passed."
                )
            }
        },

        "required": [

            "candidate_name",

            "seat_number",

            "mother_name",

            "percentage",

            "passing_year"
        ]
    }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are a document understanding system.

The following text was extracted from a 10th standard
marksheet using Google Cloud Enterprise Document OCR.

Extract these fields:

1. candidate_name
2. seat_number
3. mother_name
4. percentage
5. passing_year

Important instructions:

- Use only information present in the OCR text.
- Do not invent values.
- Do not calculate values.
- Do not correct names.
- Preserve values as written.
- If a field cannot be identified, return an empty string.
- Return only the requested structured fields.

OCR TEXT:

------------------------------
{ocr_text}
------------------------------
"""

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

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
        "Gemini client initialized"
    )

    # --------------------------------------------------------
    # Gemini request
    # --------------------------------------------------------

    logger.info(
        "Sending OCR text to Gemini 2.5 Pro..."
    )

    gemini_api_start = (
        time.perf_counter()
    )

    response = (
        client.models.generate_content(

            model=GEMINI_MODEL,

            contents=prompt,

            config={

                "response_mime_type":
                    "application/json",

                "response_schema":
                    response_schema,

                "temperature": 0
            }
        )
    )

    gemini_api_time = (

        time.perf_counter()

        - gemini_api_start
    )

    logger.info(
        f"Gemini API response received "
        f"in {gemini_api_time:.3f} seconds"
    )

    # --------------------------------------------------------
    # Print formatted JSON
    # --------------------------------------------------------

    print(
        "\n========== GEMINI RESULT ==========\n"
    )

    try:

        parsed_result = json.loads(
            response.text
        )

        print(
            json.dumps(
                parsed_result,
                indent=4,
                ensure_ascii=False
            )
        )

    except json.JSONDecodeError:

        logger.error(
            "Gemini response is not valid JSON."
        )

        print(
            response.text
        )

    print(
        "\n========== END GEMINI RESULT ==========\n"
    )

    # --------------------------------------------------------
    # Total Gemini time
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
    logger.info(
        "GEMINI 2.5 PRO EXTRACTION FINISHED"
    )
    logger.info("------------------------------------------")

    return response.text


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    start_time = (
        time.perf_counter()
    )

    logger.info("------------------------------------------")
    logger.info(
        "SAVING RESULT"
    )
    logger.info("------------------------------------------")

    output_path = Path(
        OUTPUT_FILE
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    logger.info(
        f"Output file: "
        f"{output_path.resolve()}"
    )

    try:

        parsed_result = json.loads(
            result
        )

    except json.JSONDecodeError as error:

        raise ValueError(
            f"Invalid Gemini JSON response: "
            f"{error}"
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

    save_time = (

        time.perf_counter()

        - start_time
    )

    logger.info(
        "RESULT SAVED SUCCESSFULLY"
    )

    logger.info(
        f"JSON SAVE TIME: "
        f"{save_time:.3f} seconds"
    )

    logger.info(
        f"Output file: "
        f"{output_path.resolve()}"
    )


# ============================================================
# PRINT FINAL JSON
# ============================================================

def print_final_json(result):

    try:

        parsed_result = json.loads(
            result
        )

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
                parsed_result,
                indent=4,
                ensure_ascii=False
            )
        )

        print(
            "=" * 60
        )

        print(
            "END FINAL JSON OUTPUT"
        )

        print(
            "=" * 60
        )

    except json.JSONDecodeError:

        logger.error(
            "Unable to format final response as JSON."
        )

        print(
            result
        )


# ============================================================
# MAIN
# ============================================================

def main():

    application_start_time = (
        time.perf_counter()
    )

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "10TH MARKSHEET DATA EXTRACTION STARTED"
    )

    logger.info(
        "=" * 60
    )

    # --------------------------------------------------------
    # Get project automatically
    # --------------------------------------------------------

    project_id = get_project_id()

    # --------------------------------------------------------
    # Get processor automatically
    # --------------------------------------------------------

    processor_id = get_processor_id(
        project_id
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    validate_input_file()

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    logger.info(
        f"Project       : {project_id}"
    )

    logger.info(
        f"Location      : {LOCATION}"
    )

    logger.info(
        f"Processor ID  : {processor_id}"
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
    # STEP 1 - OCR
    # ========================================================

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "STEP 1 - ENTERPRISE DOCUMENT OCR"
    )

    logger.info(
        "=" * 60
    )

    step1_start = (
        time.perf_counter()
    )

    document = run_enterprise_ocr(

        project_id,

        processor_id,

        INPUT_FILE
    )

    step1_time = (

        time.perf_counter()

        - step1_start
    )

    logger.info(
        f"STEP 1 - Enterprise OCR: "
        f"{step1_time:.3f} seconds"
    )

    # ========================================================
    # STEP 2 - GEMINI
    # ========================================================

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "STEP 2 - GEMINI 2.5 PRO"
    )

    logger.info(
        "=" * 60
    )

    step2_start = (
        time.perf_counter()
    )

    result = extract_with_gemini(

        project_id,

        document
    )

    step2_time = (

        time.perf_counter()

        - step2_start
    )

    logger.info(
        f"STEP 2 - Gemini 2.5 Pro: "
        f"{step2_time:.3f} seconds"
    )

    # ========================================================
    # STEP 3 - SAVE JSON
    # ========================================================

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "STEP 3 - SAVE JSON"
    )

    logger.info(
        "=" * 60
    )

    step3_start = (
        time.perf_counter()
    )

    save_result(
        result
    )

    step3_time = (

        time.perf_counter()

        - step3_start
    )

    logger.info(
        f"STEP 3 - Save JSON: "
        f"{step3_time:.3f} seconds"
    )

    # ========================================================
    # STEP 4 - FINAL JSON
    # ========================================================

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "STEP 4 - FINAL JSON OUTPUT"
    )

    logger.info(
        "=" * 60
    )

    print_final_json(
        result
    )

    # ========================================================
    # TOTAL TIME
    # ========================================================

    total_execution_time = (

        time.perf_counter()

        - application_start_time
    )

    logger.info("")
    logger.info(
        "=" * 60
    )

    logger.info(
        "PERFORMANCE SUMMARY"
    )

    logger.info(
        "=" * 60
    )

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

    logger.info(
        "=" * 60
    )

    logger.info(
        "10TH MARKSHEET DATA EXTRACTION FINISHED"
    )

    logger.info(
        "=" * 60
    )

    logger.info(
        f"JSON file available at: "
        f"{Path(OUTPUT_FILE).resolve()}"
    )


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