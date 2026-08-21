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

PROCESSOR_VERSION = (
    "pretrained-foundation-model-v1.5-2025-08-06"
)


# ============================================================
# GEMINI CONFIGURATION
# ============================================================

GEMINI_MODEL = "gemini-2.5-flash-lite"

GEMINI_LOCATION = "global"


# ============================================================
# FILE CONFIGURATION
# ============================================================

DOCUMENT_PATH = "../docs/noncreamyLayerCertificates/non-creamyLayer_2.jpg"

OUTPUT_FILE = "output/non_creamy_layer_result.json"


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
        ".tiff": "image/tiff",
    }

    if extension not in mime_types:

        raise ValueError(
            f"Unsupported file format: {extension}"
        )

    return mime_types[extension]


# ============================================================
# ENTERPRISE DOCUMENT AI OCR
# ============================================================

def run_document_ai_ocr(document_path):

    total_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info("ENTERPRISE DOCUMENT AI OCR STARTED")
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Configuration logs
    # --------------------------------------------------------

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
        "Input file: %s",
        document_path
    )

    # --------------------------------------------------------
    # Check document
    # --------------------------------------------------------

    if not os.path.isfile(document_path):

        raise FileNotFoundError(
            f"Document not found: "
            f"{os.path.abspath(document_path)}"
        )

    logger.info(
        "Document found successfully"
    )

    # --------------------------------------------------------
    # MIME type
    # --------------------------------------------------------

    mime_type = get_mime_type(
        document_path
    )

    logger.info(
        "MIME type: %s",
        mime_type
    )

    # --------------------------------------------------------
    # Create Document AI client
    # --------------------------------------------------------

    client_start = time.perf_counter()

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
        - client_start
    )

    logger.info(
        "Document AI client initialized in %.3f seconds",
        client_time
    )

    # --------------------------------------------------------
    # Processor resource
    # --------------------------------------------------------

    processor_name = (
        f"projects/{PROJECT_ID}"
        f"/locations/{LOCATION}"
        f"/processors/{PROCESSOR_ID}"
        f"/processorVersions/{PROCESSOR_VERSION}"
    )

    logger.info(
        "Processor resource: %s",
        processor_name
    )

    # --------------------------------------------------------
    # Read document
    # --------------------------------------------------------

    read_start = time.perf_counter()

    with open(
        document_path,
        "rb"
    ) as file:

        document_content = file.read()

    read_time = (
        time.perf_counter()
        - read_start
    )

    logger.info(
        "Document read time: %.3f seconds",
        read_time
    )

    logger.info(
        "Document size: %.3f MB",
        len(document_content) / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Raw document
    # --------------------------------------------------------

    raw_document = documentai.RawDocument(
        content=document_content,
        mime_type=mime_type
    )

    # --------------------------------------------------------
    # OCR configuration
    # --------------------------------------------------------

    process_options = documentai.ProcessOptions(
        ocr_config=documentai.OcrConfig(
            enable_native_pdf_parsing=True,
            enable_image_quality_scores=True,
            enable_symbol=True
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
    # Send to Enterprise Document AI
    # --------------------------------------------------------

    logger.info("")
    logger.info(
        "Sending Non-Creamy Layer certificate "
        "to Enterprise Document AI..."
    )

    ocr_start = time.perf_counter()

    response = client.process_document(
        request=request
    )

    ocr_time = (
        time.perf_counter()
        - ocr_start
    )

    logger.info(
        "Enterprise Document AI response received"
    )

    logger.info(
        "Enterprise OCR API time: %.3f seconds",
        ocr_time
    )

    # --------------------------------------------------------
    # OCR result
    # --------------------------------------------------------

    document = response.document

    ocr_text = document.text

    page_count = len(
        document.pages
    )

    logger.info(
        "Pages detected: %d",
        page_count
    )

    logger.info(
        "OCR text length: %d characters",
        len(ocr_text)
    )

    # --------------------------------------------------------
    # Display OCR text
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("ENTERPRISE OCR TEXT")
    print("=" * 70)

    print(ocr_text)

    print("=" * 70)

    total_ocr_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        "TOTAL ENTERPRISE OCR TIME: %.3f seconds",
        total_ocr_time
    )

    logger.info("=" * 70)
    logger.info("ENTERPRISE DOCUMENT AI OCR FINISHED")
    logger.info("=" * 70)

    return ocr_text


# ============================================================
# GEMINI 2.5 FLASH-LITE
# ============================================================

def extract_with_gemini(ocr_text):

    total_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "GEMINI 2.5 FLASH-LITE EXTRACTION STARTED"
    )
    logger.info("=" * 70)

    logger.info(
        "Gemini model: %s",
        GEMINI_MODEL
    )

    logger.info(
        "OCR text length sent to Gemini: %d characters",
        len(ocr_text)
    )

    # --------------------------------------------------------
    # Structured output schema
    # --------------------------------------------------------

    response_schema = {
        "type": "OBJECT",

        "properties": {

            "document_type": {
                "type": "STRING"
            },

            "certificate_number": {
                "type": "STRING"
            },

            "application_number": {
                "type": "STRING"
            },

            "applicant_name": {
                "type": "STRING"
            },

            "father_name": {
                "type": "STRING"
            },

            "mother_name": {
                "type": "STRING"
            },

            "date_of_birth": {
                "type": "STRING"
            },

            "gender": {
                "type": "STRING"
            },

            "caste": {
                "type": "STRING"
            },

            "category": {
                "type": "STRING"
            },

            "community": {
                "type": "STRING"
            },

            "non_creamy_layer_status": {
                "type": "STRING"
            },

            "financial_year": {
                "type": "STRING"
            },

            "annual_family_income": {
                "type": "STRING"
            },

            "income_limit": {
                "type": "STRING"
            },

            "income_certificate_number": {
                "type": "STRING"
            },

            "residence_address": {
                "type": "STRING"
            },

            "village": {
                "type": "STRING"
            },

            "taluka": {
                "type": "STRING"
            },

            "district": {
                "type": "STRING"
            },

            "state": {
                "type": "STRING"
            },

            "pincode": {
                "type": "STRING"
            },

            "issue_date": {
                "type": "STRING"
            },

            "valid_from": {
                "type": "STRING"
            },

            "valid_to": {
                "type": "STRING"
            },

            "issuing_authority": {
                "type": "STRING"
            },

            "designation": {
                "type": "STRING"
            },

            "office": {
                "type": "STRING"
            },

            "aadhaar_number": {
                "type": "STRING"
            },

            "other_id_number": {
                "type": "STRING"
            },

            "remarks": {
                "type": "STRING"
            }
        },

        "required": [
            "document_type",
            "certificate_number",
            "application_number",
            "applicant_name",
            "father_name",
            "mother_name",
            "date_of_birth",
            "gender",
            "caste",
            "category",
            "community",
            "non_creamy_layer_status",
            "financial_year",
            "annual_family_income",
            "income_limit",
            "income_certificate_number",
            "residence_address",
            "village",
            "taluka",
            "district",
            "state",
            "pincode",
            "issue_date",
            "valid_from",
            "valid_to",
            "issuing_authority",
            "designation",
            "office",
            "aadhaar_number",
            "other_id_number",
            "remarks"
        ]
    }

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an enterprise document understanding system.

The following text was extracted using
Google Cloud Enterprise Document AI OCR.

The document is an Indian
NON-CREAMY LAYER CERTIFICATE.

The certificate may be written in:

- English
- Marathi
- Hindi
- A combination of these languages.

Your task is to extract the following information:

1. document_type
2. certificate_number
3. application_number
4. applicant_name
5. father_name
6. mother_name
7. date_of_birth
8. gender
9. caste
10. category
11. community
12. non_creamy_layer_status
13. financial_year
14. annual_family_income
15. income_limit
16. income_certificate_number
17. residence_address
18. village
19. taluka
20. district
21. state
22. pincode
23. issue_date
24. valid_from
25. valid_to
26. issuing_authority
27. designation
28. office
29. aadhaar_number
30. other_id_number
31. remarks

IMPORTANT INSTRUCTIONS:

- Extract information only from the OCR text.
- Do not invent information.
- Do not guess missing values.
- If a field is not present, return an empty string.
- Do not calculate any value.
- Do not calculate family income.
- Do not calculate income limits.
- Do not assume the applicant's caste.
- Do not assume the applicant's category.
- Do not assume Non-Creamy Layer status.
- Extract Non-Creamy Layer status only when it is
  explicitly stated or clearly supported by the certificate.
- Preserve names as they appear in the certificate.
- Preserve certificate numbers exactly as written.
- Preserve application numbers exactly as written.
- Preserve dates exactly as they appear whenever possible.
- Preserve income values exactly as written.
- Do not modify caste names.
- Do not convert category names unless the certificate
  explicitly provides the category.
- Do not confuse caste certificate information with
  Non-Creamy Layer information.
- Do not confuse annual family income with the
  Non-Creamy Layer income limit.
- If an income limit is mentioned, extract it as
  income_limit.
- If an actual family income is mentioned, extract it as
  annual_family_income.
- Do not calculate years or financial years.
- If a financial year is explicitly mentioned, extract it.
- Do not reconstruct missing Aadhaar digits.
- Do not generate any identification number.
- Do not confuse applicant address with issuing authority
  office address.
- Extract village, taluka, district, state and pincode
  separately whenever explicitly available.
- Extract issuing authority and designation separately.
- If there is a QR code but its contents are not present
  in the OCR text, do not invent its data.
- If a field is uncertain or unavailable, return an
  empty string.

IMPORTANT FOR INDIAN DOCUMENTS:

The same field may appear under different labels.

Applicant name may appear as:

- Name
- Applicant Name
- Beneficiary Name
- अर्जदाराचे नाव
- अर्जदाराचे पूर्ण नाव
- नाम

Father name may appear as:

- Father Name
- Father's Name
- S/o
- Son of
- वडिलांचे नाव
- पिता का नाम

Mother name may appear as:

- Mother Name
- Mother's Name
- आईचे नाव
- माता का नाम

Certificate number may appear as:

- Certificate No.
- Certificate Number
- प्रमाणपत्र क्रमांक
- दाखला क्रमांक
- प्रमाणपत्र क्र.

Application number may appear as:

- Application No.
- Application Number
- अर्ज क्रमांक
- अर्ज संख्या

Caste may appear as:

- Caste
- जात
- जात प्रवर्ग
- Caste Name

Category may appear as:

- Category
- वर्ग
- प्रवर्ग
- Category of Applicant

Non-Creamy Layer may appear as:

- Non-Creamy Layer
- Non Creamy Layer
- NCL
- Non-Creamy Layer Certificate
- नॉन क्रिमीलेयर
- नॉन-क्रीमी लेयर
- नॉन क्रिमिलेयर
- Creamy Layer / Non-Creamy Layer

Income may appear as:

- Annual Income
- Family Income
- Annual Family Income
- उत्पन्न
- वार्षिक उत्पन्न
- कुटुंबाचे वार्षिक उत्पन्न

Issuing authority may appear as:

- Tahsildar
- Tehsildar
- Naib Tahsildar
- Revenue Officer
- तहसीलदार
- नायब तहसीलदार
- महसूल अधिकारी

Do not treat these labels as literal field values.
Extract the actual information associated with them.

OCR TEXT:

--------------------------------------------------
{ocr_text}
--------------------------------------------------

Return only the requested JSON structure.
"""

    # --------------------------------------------------------
    # Gemini client
    # --------------------------------------------------------

    logger.info(
        "Initializing Gemini Vertex AI client..."
    )

    client_start = time.perf_counter()

    client = genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=GEMINI_LOCATION,
        http_options=HttpOptions(
            api_version="v1"
        )
    )

    client_time = (
        time.perf_counter()
        - client_start
    )

    logger.info(
        "Gemini client initialized in %.3f seconds",
        client_time
    )

    # --------------------------------------------------------
    # Gemini API
    # --------------------------------------------------------

    logger.info(
        "Sending OCR text to Gemini 2.5 Flash-Lite..."
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

    gemini_api_time = (
        time.perf_counter()
        - gemini_start
    )

    logger.info(
        "Gemini API response received in %.3f seconds",
        gemini_api_time
    )

    # --------------------------------------------------------
    # Parse response
    # --------------------------------------------------------

    result = json.loads(
        response.text
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("")
    print("=" * 70)
    print("GEMINI 2.5 FLASH-LITE RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=4,
            ensure_ascii=False
        )
    )

    print("=" * 70)

    total_gemini_time = (
        time.perf_counter()
        - total_start
    )

    logger.info(
        "TOTAL GEMINI EXTRACTION TIME: %.3f seconds",
        total_gemini_time
    )

    logger.info("=" * 70)
    logger.info(
        "GEMINI 2.5 FLASH-LITE EXTRACTION FINISHED"
    )
    logger.info("=" * 70)

    return result


# ============================================================
# SAVE RESULT
# ============================================================

def save_result(result):

    start_time = time.perf_counter()

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

    save_time = (
        time.perf_counter()
        - start_time
    )

    logger.info(
        "JSON saved successfully"
    )

    logger.info(
        "Output file: %s",
        os.path.abspath(OUTPUT_FILE)
    )

    logger.info(
        "JSON save time: %.3f seconds",
        save_time
    )

    return save_time


# ============================================================
# MAIN
# ============================================================

def main():

    application_start = time.perf_counter()

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "NON-CREAMY LAYER CERTIFICATE "
        "EXTRACTION PIPELINE STARTED"
    )
    logger.info("=" * 70)

    try:

        # ====================================================
        # STEP 1 - ENTERPRISE OCR
        # ====================================================

        step1_start = time.perf_counter()

        ocr_text = run_document_ai_ocr(
            DOCUMENT_PATH
        )

        step1_time = (
            time.perf_counter()
            - step1_start
        )

        logger.info(
            "STEP 1 - Enterprise OCR: %.3f seconds",
            step1_time
        )

        # ====================================================
        # STEP 2 - GEMINI
        # ====================================================

        step2_start = time.perf_counter()

        final_result = extract_with_gemini(
            ocr_text
        )

        step2_time = (
            time.perf_counter()
            - step2_start
        )

        logger.info(
            "STEP 2 - Gemini 2.5 Flash-Lite: %.3f seconds",
            step2_time
        )

        # ====================================================
        # STEP 3 - SAVE
        # ====================================================

        step3_start = time.perf_counter()

        save_time = save_result(
            final_result
        )

        step3_time = (
            time.perf_counter()
            - step3_start
        )

        logger.info(
            "STEP 3 - Save JSON: %.3f seconds",
            step3_time
        )

        # ====================================================
        # TOTAL PIPELINE TIME
        # ====================================================

        total_time = (
            time.perf_counter()
            - application_start
        )

        # ====================================================
        # FINAL PERFORMANCE SUMMARY
        # ====================================================

        logger.info("")
        logger.info("=" * 70)
        logger.info("PERFORMANCE SUMMARY")
        logger.info("=" * 70)

        logger.info(
            "Enterprise OCR Time       : %.3f seconds",
            step1_time
        )

        logger.info(
            "Gemini Flash-Lite Time    : %.3f seconds",
            step2_time
        )

        logger.info(
            "JSON Save Time            : %.3f seconds",
            step3_time
        )

        logger.info(
            "-" * 70
        )

        logger.info(
            "TOTAL PIPELINE TIME       : %.3f seconds",
            total_time
        )

        logger.info(
            "TOTAL PIPELINE TIME       : %.2f minutes",
            total_time / 60
        )

        logger.info("=" * 70)

        # ====================================================
        # FINAL OUTPUT
        # ====================================================

        print("")
        print("=" * 70)
        print("FINAL NON-CREAMY LAYER CERTIFICATE JSON")
        print("=" * 70)

        print(
            json.dumps(
                final_result,
                indent=4,
                ensure_ascii=False
            )
        )

        print("=" * 70)

        print(
            f"Saved to: {os.path.abspath(OUTPUT_FILE)}"
        )

        print("=" * 70)

        logger.info(
            "NON-CREAMY LAYER CERTIFICATE EXTRACTION "
            "FINISHED SUCCESSFULLY"
        )

    except Exception as error:

        logger.exception(
            "Non-Creamy Layer certificate extraction failed: %s",
            error
        )

        raise


# ============================================================
# APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()