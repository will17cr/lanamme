# %% [code]
# coding: utf-8

# # Procesamiento informes

# ## Instalaciones

# Tardará un poco pues son pesadas

import os

os.system('pip install -U pip')

os.system('pip install sickle')

os.system('pip install scythe')

os.system('pip install oaipmh-scythe')

os.system('pip install dateparser')

os.system('pip install openpyxl')

os.system('pip install python-dateutil')

os.system('pip install python-dotenv')

os.system('pip install gspread')

os.system('pip install gspread-pandas')

os.system('pip install gspread_dataframe')

os.system('pip install google-api-python-client')

os.system('pip install keyring')

os.system('pip install selenium')

# # os.system('pip install --upgrade nbconvert')

# ## Instalar librerías para Gemini y procesar PDF

os.system('pip install google-generativeai')

os.system('pip install requests')

os.system('pip install pdfplumber')

os.system('pip install pymupdf')

os.system('pip install ocrmypdf')

os.system('pip install "img2pdf<0.6.2"')

# os.system('pip install glymur')

os.system('pip install pytesseract Pillow') # For OCR

# # Procesamiento informes LANAMME desde repositorio
# # Wilmer Ramirez Morera
# # wilmer.ramirez@gmail.com
# # wilmer.ramirez@cgr.go.cr

print(f"\n{'*'*72}\nScript to request records from LANAMME's repository and update our data\n{'*'*72}\n")

# ## -------- Importaciones --------

import datetime as dt

import os
import unicodedata
import pandas as pd
# from google.colab import drive
from ctypes.util import find_library
# import pdfminer
# from pdfminer.high_level import extract_text
# import spacy
# import es_core_news_mdz
# import es_core_news_lg

from pandas.core.dtypes.inference import is_number
from pandas.core.dtypes.common import is_numeric_v_string_like
from pdfminer.utils import isnumber

import numpy as np
import re
import time # For potential delays between API calls

from datetime import datetime
import json

from dateutil import tz

os.getcwd()

os.chdir("../")

os.getcwd()

# ### OAI-PHM

from sickle import Sickle

from oaipmh_scythe import Scythe

from pprint import pprint

# ### Google Sheets

import gspread

try:
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
except ImportError:
    os.system('pip install gspread_dataframe')
    from gspread_dataframe import set_with_dataframe, get_as_dataframe

# ### Kaggle or local

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Interactive', 'Batch'):
    from kaggle_secrets import UserSecretsClient
else:
    from dotenv import load_dotenv, find_dotenv
    _ = load_dotenv(find_dotenv()) # read local .env file

# ### Gemini API

import google.generativeai as genai

# ### PDF Processing

import requests
import pdfplumber
from io import BytesIO
import fitz
import ocrmypdf

import urllib.request
import urllib.error
from urllib.parse import urljoin # Already there, but ensure it's available
import traceback # For detailed error logging

import pytesseract # For OCR
from PIL import Image # For OCR with Tesseract

# ### Web scrapping

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, ElementNotInteractableException

from bs4 import BeautifulSoup

# ---

# ## -------- CONFIGURATION -------- 

# ### timezone

CR_TZ = tz.gettz('America/Costa_Rica')

# ### DSpace URL

URL = 'https://www.lanamme.ucr.ac.cr/oai/request?'

# ### DSpace date constraint

DSpace_start='2024-01-01'

# ### Google Sheets - Master sheet name

MASTER_SHEET_NAME = 'Master' 

# ### risk categories

RISK_CATEGORIES = ["ninguno", "bajo", "medio", "alto", "critico"] # Define globally for validation

# ### Gemini date constraint

GEMINI_PROCESSING_CUTOFF_DATE = datetime(2024, 1, 1) 

# ### Gemini specification model

# GEMINI_MODEL_NAME = 'gemini-3-flash' # Or your preferred model

# GEMINI_MODEL_NAME = 'gemini-2.5-flash-lite' # Or your preferred model

GEMINI_MODEL_NAME = 'gemini-2.5-flash' # Or your preferred model

# GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-09-2025' # Or your preferred model

# GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-05-20' # Or your preferred model

# GEMINI_MODEL_NAME = 'gemini-2.5-pro-preview-03-25' # Or your preferred model

# GEMINI_MODEL_NAME = 'gemini-2.5-pro-exp-03-25' # Or your preferred model

# ### Secrets, API keys

if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Interactive', 'Batch'):
    user_secrets = UserSecretsClient()
    try:
        GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY") # Store your Gemini API Key as a Kaggle Secret
        GOOGLE_API_SECRET = user_secrets.get_secret("Google_API")
        GOOGLE_SHEET_ID = user_secrets.get_secret("NewSheetID")
    except Exception as e:
        print(f"Error accessing Kaggle Secrets: {e}")
        print("Please ensure GEMINI_API_KEY, Google_API, and SheetID are set in Kaggle Secrets.")
        GEMINI_API_KEY = None # Or handle as a fatal error
        GOOGLE_API_SECRET = None
        GOOGLE_SHEET_ID = None
else:
    try:
        GEMINI_API_KEY = os.environ["GEMINI_API_KEY"] # Store your Gemini API Key as a Kaggle Secret
        GOOGLE_API_SECRET = os.environ["Google_API"]
        GOOGLE_SHEET_ID = os.environ["NewSheetID"]
    except Exception as e:
        print(f"Error accessing Local Secrets: {e}")
        print("Please ensure GEMINI_API_KEY, Google_API, and SheetID are set in Kaggle Secrets.")
        GEMINI_API_KEY = None # Or handle as a fatal error
        GOOGLE_API_SECRET = None
        GOOGLE_SHEET_ID = None

# ### Gemini Model

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY) 
    model = genai.GenerativeModel(GEMINI_MODEL_NAME)
else:
    model = None
    print("Gemini API Key not found. AI processing will be skipped.")

# ### Checking gemini models

def gemini_models(API_KEY=None):
    if API_KEY: 
        genai.configure(api_key=API_KEY)
        print("Gemini API Key configured.")

        print("\nAvailable models that support 'generateContent':")
        model_found = False
        for m in genai.list_models():
            # We are interested in models that can generate text content
            if 'generateContent' in m.supported_generation_methods:
                print(f"  Model name: {m.name}")
                print(f"    Display name: {m.display_name}")
                print(f"    Description: {m.description}")
                # You might also want to check m.version, etc.
                print("-" * 20)
                model_found = True

        if not model_found:
            print("No models found that support 'generateContent'. Check your API key and permissions.")

    else:
        print("GEMINI_API_KEY not found. Cannot list models.")
    return

# gemini_models(GEMINI_API_KEY)

# ## -------- HELPER FUNCTIONS --------

# ### enhanced request

def fetch_pdf_bytes_with_urllib(initial_url, max_redirects_ignored=5, timeout=45): # max_redirects is handled by urlopen
    current_url = initial_url
    # Standard browser User-Agent
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

    try:
        print(f"  urllib: Attempting to GET (with auto-redirects): {current_url}")
        req = urllib.request.Request(current_url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            # urlopen handles redirects by default.
            # Get the final URL after all redirects were followed.
            final_url = response.geturl()
            print(f"  urllib: Final URL after redirects: {final_url}")
            print(f"  urllib: Status code: {response.getcode()}")

            # Check content type from the final response
            info = response.info() # This is an http.client.HTTPMessage object
            content_type = info.get('Content-Type', '').lower()

            # if 'application/pdf' not in content_type:
            #     print(f"  urllib: ERROR - Content-Type is '{content_type}', not 'application/pdf'. Final URL: {final_url}")
            #     return None

            # ### CHANGE HERE: Logic to accept .crdownload or forced handling ###
            is_crdownload = '.crdownload' in final_url.lower() or '.crdownload' in current_url.lower()
            
            # If it's NOT a pdf AND NOT a crdownload, we reject it.
            # But if it IS a crdownload, we let it pass through.
            if 'application/pdf' not in content_type and not is_crdownload:
                print(f"  urllib: ERROR - Content-Type is '{content_type}', not 'application/pdf'. Final URL: {final_url}")
                return None
            
            if is_crdownload:
                print(f"  urllib: WARNING - .crdownload detected. Ignoring Content-Type '{content_type}' and forcing download.")
            # ###############################################################
            
            pdf_bytes = response.read()
            print(f"  urllib: PDF downloaded successfully from {final_url}. Size: {len(pdf_bytes)} bytes.")
            return pdf_bytes

    except urllib.error.HTTPError as e:
        # This catches 4xx, 5xx errors. 3xx should have been handled by urlopen.
        # If a redirect chain itself leads to an error, it might surface here.
        print(f"  urllib: HTTPError for '{current_url}': Code: {e.code}, Reason: {e.reason}")
        if hasattr(e, 'headers'):
            print(f"    Response Headers from error: {e.headers}")
        return None
    except urllib.error.URLError as e:
        # This can be a timeout (socket.timeout is often wrapped here), connection error, DNS error, etc.
        print(f"  urllib: URLError for '{current_url}': Reason: {e.reason}")
        # Check if it's a timeout
        if isinstance(e.reason, TimeoutError) or (hasattr(e.reason, 'errno') and e.reason.errno == 60): # errno 60 is ETIMEDOUT on some systems
             print(f"  urllib: Explicit timeout detected for {current_url}")
        return None
    except TimeoutError: # Catch global TimeoutError as well, socket.timeout might raise this directly
        print(f"  urllib: Global TimeoutError for {current_url}")
        return None
    except Exception as e_generic:
        print(f"  urllib: Generic error for '{current_url}': {type(e_generic).__name__} - {e_generic}")
        print(traceback.format_exc())
        return None

# ### Clean text for LLM

def clean_text_for_llm(text: str) -> str:
    if not isinstance(text, str) or not text:
        return ""

    # 1. Normalize Unicode characters
    # NFKC is a good choice: applies compatibility decomposition, followed by canonical composition.
    # It can help with ligatures and visually similar characters.
    try:
        text = unicodedata.normalize('NFKC', text)
    except Exception as e:
        print(f"Warning: Unicode normalization failed - {e}")
        # Continue with the original text if normalization fails

    # 2. Replace common problematic sequences or characters (optional, add as needed)
    #    Example: Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    #    Example: Remove specific known bad characters if you identify them
    #    text = text.replace('\ufeff', '') # Remove BOM (Byte Order Mark) if it appears mid-string

    # 3. Filter out most control characters but keep essential whitespace
    cleaned_chars = []
    for char in text:
        # Get the Unicode category of the character
        category = unicodedata.category(char)
        # Keep letters, numbers, punctuation, symbols, and common whitespace (space, newline, tab)
        if category.startswith('L') or \
           category.startswith('N') or \
           category.startswith('P') or \
           category.startswith('S') or \
           char in (' ', '\n', '\t', '\r'): # Explicitly keep common whitespace
            cleaned_chars.append(char)
        # Optionally, replace other control characters or unwanted categories with a space or remove them
        # For example, to replace other control chars ('Cc', 'Cf', 'Co', 'Cs', 'Cn') with nothing:
        # elif category.startswith('C'):
        #     pass # Discard
        # To replace with a space (be careful not to add too many spaces):
        # elif category.startswith('C'):
        #    if cleaned_chars and cleaned_chars[-1] != ' ': # Avoid double spaces
        #        cleaned_chars.append(' ')

    text = "".join(cleaned_chars)

    # 4. Ensure the text is valid UTF-8 by encoding and decoding
    # 'replace' will insert the U+FFFD (�) replacement character for any bytes
    # that can't be decoded. 'ignore' would just drop them.
    # This step primarily handles issues if the string somehow contains
    # byte sequences that are not valid UTF-8, which can happen with OCR.
    try:
        text = text.encode('utf-8', 'replace').decode('utf-8')
    except Exception as e:
        print(f"Warning: UTF-8 encode/decode cleaning step failed - {e}")
        # Fallback or decide how to handle, for now, use the text as is from previous steps

    return text.strip()

# ### Get pdf URL

def get_pdf_url(url):

    options = webdriver.FirefoxOptions()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    # options.add_argument("--window-size=1920,1080")

    web=f"{url}?show=full"

    browser = webdriver.Firefox(options=options)

    if not url or not isinstance(url, str) or not url.startswith('http'):
        print(f"    get_direct_pdf_url: Invalid page_url: '{url}'")
        return None

    browser.get(web)
    time.sleep(250 / 1000)

    html = browser.page_source
    pagina = BeautifulSoup(html, 'html.parser')

    pdf_url_tag = pagina.find('meta', {'name': 'citation_pdf_url'})
    if pdf_url_tag:
        pdf_url = pdf_url_tag['content']
    else:
        return None

    browser.close()
    browser.stop_client()

    return pdf_url

# ### Extract PDF

# #### basic

def extract_text_from_pdf_url(pdf_url):
    """Fetches a PDF from a URL and extracts text content."""
    if not pdf_url or not isinstance(pdf_url, str) or not pdf_url.lower().endswith('.pdf'):
        print(f"Invalid or non-PDF URL: {pdf_url}")
        return None
    try:
        response = requests.get(pdf_url, timeout=30) # Added timeout
        response.raise_for_status()  # Raises an exception for bad status codes

        text_content = ""
        with BytesIO(response.content) as pdf_file:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        return text_content.strip() if text_content else None
    except requests.exceptions.RequestException as e:
        print(f"Error downloading PDF from {pdf_url}: {e}")
    except Exception as e:
        print(f"Error extracting text from PDF {pdf_url}: {e}")
    return None

# #### Fitz

def extract_text_from_pdf_fitz(pdf_url_direct_initial):
    if not pdf_url_direct_initial or not isinstance(pdf_url_direct_initial, str) or not pdf_url_direct_initial.lower().startswith('http'):
        print(f"  Fitz: Error - Invalid URL format for PDF extraction: '{pdf_url_direct_initial}'")
        return None

    if not pdf_url_direct_initial.lower().endswith('.pdf'):
        print(f"  Fitz: Warning - Direct PDF URL does not end with .pdf: '{pdf_url_direct_initial}'. Attempting download anyway.")

    print(f"  Fitz: Attempting download via urllib for: {pdf_url_direct_initial}")
    pdf_bytes = fetch_pdf_bytes_with_urllib(pdf_url_direct_initial, timeout=45) # Pass timeout

    if not pdf_bytes:
        print(f"  Fitz: Failed to download PDF bytes using urllib for {pdf_url_direct_initial}.")
        return None

    print(f"  Fitz: PDF bytes received ({len(pdf_bytes)} bytes). Processing with PyMuPDF...")
    text_content = ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        # print(f"  Fitz: Extracting text from {len(doc)} page(s)...") # Log is now in the fetcher
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text("text")
            if page_text:
                text_content += page_text + "\n"
        doc.close()
        if text_content.strip():
            print(f"  Fitz: Text extraction complete.")
        else:
            print(f"  Fitz: No text could be extracted (PDF might be image-based or empty of text).")
        return text_content.strip() if text_content.strip() else None
    except Exception as e_fitz_proc:
        print(f"  Fitz: Error opening or parsing PDF content with PyMuPDF from '{pdf_url_direct_initial}': {e_fitz_proc}")
        print(traceback.format_exc())
        return None

# #### pytesseract

def extract_text_from_pdf_ocr(pdf_url_direct, lang_code='spa'):
    if not pdf_url_direct or not isinstance(pdf_url_direct, str) or not pdf_url_direct.lower().startswith('http'):
        print(f"  OCR (Pytesseract): Error - Invalid URL for PDF: '{pdf_url_direct}'")
        return None

    try:
        print(f"  OCR (Pytesseract): Attempting download via urllib for OCR: {pdf_url_direct}")
        # --- USE THE URLLIB FETCHER ---
        pdf_bytes = fetch_pdf_bytes_with_urllib(pdf_url_direct, timeout=45)
        # -----------------------------

        if not pdf_bytes:
            print(f"  OCR (Pytesseract): Failed to download PDF using urllib for '{pdf_url_direct}'.")
            return None

        print(f"  OCR (Pytesseract): PDF downloaded via urllib for OCR. Size: {len(pdf_bytes)} bytes.")
        all_ocr_text = ""
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            # ... (rest of your Pytesseract logic using doc.load_page, page.get_pixmap, Image.open, pytesseract.image_to_string) ...
            print(f"  OCR (Pytesseract): Processing {len(doc)} pages with Tesseract (lang: {lang_code})...")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300) 
                img_bytes_pil = pix.tobytes("png") # Changed variable name to avoid conflict with pdf_bytes

                try:
                    pil_image = Image.open(BytesIO(img_bytes_pil))
                    page_ocr_text = pytesseract.image_to_string(pil_image, lang=lang_code)
                    if page_ocr_text:
                        all_ocr_text += page_ocr_text + "\n\n"
                    print(f"    OCR (Pytesseract): Page {page_num + 1} processed.")
                except Exception as e_ocr_page:
                    print(f"    OCR (Pytesseract): Error processing page {page_num + 1} with Tesseract: {e_ocr_page}")
            # ... (logging and return) ...
            if all_ocr_text.strip():
                print("  OCR (Pytesseract): Text extraction via OCR complete.")
            else:
                print("  OCR (Pytesseract): No text extracted via OCR.")
            return all_ocr_text.strip() if all_ocr_text.strip() else None

        except Exception as e_ocr_fitz_setup: # Error opening PDF with fitz before OCR
            print(f"  OCR (Pytesseract): Error opening PDF with PyMuPDF for OCR preparation: {e_ocr_fitz_setup}")
            print(traceback.format_exc())
            return None
        finally:
            if doc:
                doc.close()

    except Exception as e_generic_download_setup: # Errors during download setup
        print(f"OCR (Pytesseract): Generic error during download/setup for OCR: {type(e_generic_download_setup).__name__} - {e_generic_download_setup}")
        print(traceback.format_exc())
        return None

# #### OCRmyPDF

import tempfile # Make sure this is at the top of your script with other imports
import os       # Make sure this is at the top of your script (it likely already is)

def extract_text_from_pdf_ocred(pdf_url_direct, lang_code='spa'):
    if not pdf_url_direct or not isinstance(pdf_url_direct, str) or not pdf_url_direct.lower().startswith('http'):
        print(f"  OCRmyPDF: Error - Invalid URL format for PDF processing: '{pdf_url_direct}'")
        return None

    # Warning for non .pdf extension is fine to keep
    if not pdf_url_direct.lower().endswith('.pdf'):
        print(f"  OCRmyPDF: Warning - Direct PDF URL does not end with .pdf: '{pdf_url_direct}'. Attempting download anyway.")

    input_pdf_path = None
    output_pdf_path = None

    try:
        print(f"  OCRmyPDF: Attempting to download direct PDF via urllib from: {pdf_url_direct}")
        # --- USE THE URLLIB FETCHER ---
        pdf_bytes_original = fetch_pdf_bytes_with_urllib(pdf_url_direct, timeout=60)
        # -----------------------------

        if not pdf_bytes_original:
            print(f"  OCRmyPDF: Failed to download PDF using urllib for '{pdf_url_direct}'.")
            return None

        print(f"  OCRmyPDF: PDF bytes downloaded via urllib. Size: {len(pdf_bytes_original)} bytes.")
        text_content = ""

        # Create a named temporary file for the input PDF
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_in:
            input_pdf_path = tmp_in.name
            tmp_in.write(pdf_bytes_original)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_out:
            output_pdf_path = tmp_out.name

        print(f"  OCRmyPDF: Input temp file: {input_pdf_path}")
        print(f"  OCRmyPDF: Output temp file: {output_pdf_path}")

        try:
            ocrmypdf.ocr(input_pdf_path, output_pdf_path,
                         language=lang_code, force_ocr=True, deskew=True,
                         optimize=2, skip_text=False, invalidate_digital_signatures=True)
            print(f"  OCRmyPDF: OCR process completed. Output saved to {output_pdf_path}")

            doc = fitz.open(output_pdf_path)
            # ... (rest of fitz text extraction from output_pdf_path) ...
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text")
                if page_text:
                    text_content += page_text + "\n"
            doc.close()
            # ... (logging and return) ...
            if text_content.strip():
                print(f"  OCRmyPDF: Text extraction complete from OCR'd PDF file.")
            else:
                print(f"  OCRmyPDF: No text could be extracted from OCR'd PDF file.")
            return text_content.strip() if text_content.strip() else None

        except ocrmypdf.exceptions.InputFileError as e: # Specific ocrmypdf errors
            print(f"  OCRmyPDF Input Error for '{pdf_url_direct}' (file {input_pdf_path}): {e}.")
            return None
        # ... (other specific ocrmypdf exceptions you had) ...
        except Exception as e_ocrmypdf_process:
            print(f"  OCRmyPDF: General error during OCR or text extraction for '{pdf_url_direct}': {type(e_ocrmypdf_process).__name__} - {e_ocrmypdf_process}")
            print(traceback.format_exc()) # Add traceback for these errors too
            return None

    except Exception as e_generic_download_setup: # Errors during download setup or initial phases
        print(f"OCRmyPDF: Generic error during download/setup for '{pdf_url_direct}': {type(e_generic_download_setup).__name__} - {e_generic_download_setup}")
        print(traceback.format_exc()) # Add traceback
        return None
    finally:
        if input_pdf_path and os.path.exists(input_pdf_path):
            try: os.remove(input_pdf_path); print(f"  OCRmyPDF: Cleaned up input temp file: {input_pdf_path}")
            except Exception as e_clean_in: print(f"  OCRmyPDF: Warning - could not delete input temp file {input_pdf_path}: {e_clean_in}")
        if output_pdf_path and os.path.exists(output_pdf_path):
            try: os.remove(output_pdf_path); print(f"  OCRmyPDF: Cleaned up output temp file: {output_pdf_path}")
            except Exception as e_clean_out: print(f"  OCRmyPDF: Warning - could not delete output temp file {output_pdf_path}: {e_clean_out}")

# ### Gemini Analysis


# def get_gemini_analysis(document_text):
#     default_error_rating = "Error: AI Analysis Failed"
#     default_error_explanation = "Error: AI Analysis Failed to provide explanation."
#     default_error_summary = "Error: AI Analysis Failed to provide summary."

#     if not model or not document_text:
#         print("    Gemini model not available or no document text provided for analysis.")
#         return default_error_rating, default_error_explanation, default_error_summary

#     prompt_combined_json = f"""
#     Analyze the following document text, which is in Spanish.
#     Based on your analysis, generate a JSON object with the following three keys:
#     1. "riesgo_rating": A single string value representing the overall risk level. Choose EXCLUSIVELY from this list: {RISK_CATEGORIES}. This rating should be based on the presence, severity, and quantity of warning/alarming statements, and the extent of any danger stated.
#     2. "riesgo_explicacion": A detailed textual explanation IN SPANISH. Start your explanation by explicitly stating the assigned 'riesgo_rating' and then detail the primary reasons for this rating, referencing specific warnings, dangers, and concerns from the document. For example: 'El riesgo se considera [valor de riesgo_rating] debido a [razones principales y detalles específicos del documento].' Ensure any special characters like backslashes or quotes within this explanation are correctly escaped for JSON string format (e.g., a backslash should be '\\\\', a quote '\\"').
#     3. "resumen_detallado_ia": A comprehensive and explanatory summary of the entire document, IN SPANISH. **This summary MUST be less than 1000 words.** It should focus on key findings, methodologies (if applicable), conclusions, and recommendations. Ensure any special characters like backslashes or quotes within this summary are correctly escaped for JSON string format.

#     Ensure the output is ONLY a valid JSON object. Do not add any text before or after the JSON object. All string values inside the JSON must be properly JSON escaped. Example format:
#     {{
#       "riesgo_rating": "bajo",
#       "riesgo_explicacion": "El riesgo se considera bajo porque solo se mencionaron algunas preocupaciones menores y no hay indicios de peligro inminente. Por ejemplo, una ruta de archivo podría ser C:\\\\Users\\\\temp.",
#       "resumen_detallado_ia": "El documento trata sobre la \\"importancia\\" de..."
#     }}

#     Document Text (Spanish):
#     ---
#     {document_text[:1500000]}
#     ---

#     JSON Output:
#     """

#     print("    Requesting combined (rating, explanation, summary) JSON from Gemini...")
#     raw_response_text_for_debug = "No response received" # For debugging

#     try:
#         # Explicitly set timeout for the API call if needed
#         # It's good that you have request_options={'timeout': 180}
#         response = model.generate_content(prompt_combined_json, request_options={'timeout': 180})

#         if not response.parts:
#             # ... (your existing block reason handling is good) ...
#             block_reason_str = "Unknown reason"
#             if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
#                 if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason is not None:
#                     block_reason_str = str(response.prompt_feedback.block_reason)
#                 elif hasattr(response.prompt_feedback, 'safety_ratings'):
#                     for rating_info in response.prompt_feedback.safety_ratings:
#                         if str(rating_info.probability) not in ["HarmProbability.NEGLIGIBLE", "HarmProbability.LOW"]:
#                            block_reason_str = f"Safety block - Category: {rating_info.category}, Probability: {rating_info.probability.name}"
#                            break
#             print(f"    Gemini API Warning: No content parts in response. Effective Block reason: {block_reason_str}.")
#             return f"AI Error: No Parts ({block_reason_str})", default_error_explanation, default_error_summary

#         raw_response_text_for_debug = response.text # Store for debugging before parsing

#         # --- ENHANCED JSON EXTRACTION LOGIC ---
#         text_to_parse = raw_response_text_for_debug.strip()
#         json_candidate_text = None

#         # 1. Try to find content within markdown code blocks (```json ... ```)
#         #    Using re.DOTALL to match across newlines
#         match_code_block = re.search(r"```json\s*(.*?)\s*```", text_to_parse, re.DOTALL)
#         if match_code_block:
#             json_candidate_text = match_code_block.group(1).strip()
#             print(f"    Extracted JSON from markdown block.")
#         else:
#             # 2. Try to find the first/outermost JSON object directly (most common fallback)
#             match_json_object = re.search(r"\{.*\}", text_to_parse, re.DOTALL)
#             if match_json_object:
#                 json_candidate_text = match_json_object.group(0).strip()
#                 print(f"    Extracted JSON using regex for outermost curly braces.")
#             else:
#                 # 3. If no clear JSON block or object, assume the whole response *should* be JSON
#                 #    and try to clean common non-JSON prefixes/suffixes. This is more risky.
#                 json_candidate_text = text_to_parse.strip()
#                 print(f"    No clear JSON block found, attempting to parse entire response.")

#         if not json_candidate_text:
#             print("    Failed to extract any JSON candidate text.")
#             return "Error: No JSON Candidate", default_error_explanation, default_error_summary

#         print(f"    JSON candidate (first 500 chars): {json_candidate_text[:500]}...")

#         # Attempt to load the JSON
#         try:
#             response_json = json.loads(json_candidate_text)

#             categorical_riesgo = response_json.get("riesgo_rating", default_error_rating).strip().lower()
#             risk_explanation = response_json.get("riesgo_explicacion", default_error_explanation).strip()
#             resumen_ia = response_json.get("resumen_detallado_ia", default_error_summary).strip()

#             if categorical_riesgo not in RISK_CATEGORIES and not categorical_riesgo.startswith("error:"): # Lowercased "Error:" for robustness
#                 print(f"    Warning: Gemini returned an invalid risk category: '{categorical_riesgo}'.")
#                 categorical_riesgo = f"Error: Invalid Category ({categorical_riesgo})"

#             return categorical_riesgo, risk_explanation, resumen_ia

#         except json.JSONDecodeError as e_json:
#             print(f"    Error decoding JSON from Gemini: {e_json}")
#             print(f"    Problematic JSON candidate text (first 1000 chars):\n{json_candidate_text[:1000]}")
#             return "Error: JSON Decode", f"JSON Error - {e_json}", f"JSON Error - {e_json}"
#         except AttributeError: # If response.text was None
#             print(f"    Error: response.text was None or attribute error from Gemini response object.")
#             return "Error: Response Attribute", default_error_explanation, default_error_summary

#     except Exception as e_api:
#         print(f"    Major Error during Gemini API call or critical response issue: {type(e_api).__name__} - {e_api}")
#         print(traceback.format_exc()) # Print full traceback for API errors
#         # Also print the raw response text if available and different from default
#         if raw_response_text_for_debug != "No response received":
#             print(f"    Raw text received before error (if any - first 1000 chars):\n{raw_response_text_for_debug[:1000]}")
#         return f"API Exception: {type(e_api).__name__}", default_error_explanation, default_error_summary

from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError
import time

def get_gemini_analysis(document_text):
    default_error_rating = "Error: AI Analysis Failed"
    default_error_explanation = "Error: AI Analysis Failed to provide explanation."
    default_error_summary = "Error: AI Analysis Failed to provide summary."

    if not model or not document_text:
        print("    Gemini model not available or no document text provided for analysis.")
        return default_error_rating, default_error_explanation, default_error_summary

    prompt_combined_json = f"""
    Analyze the following document text, which is in Spanish.
    Based on your analysis, generate a JSON object with the following three keys:
    1. "riesgo_rating": A single string value representing the overall risk level. Choose EXCLUSIVELY from this list: {RISK_CATEGORIES}. This rating should be based on the presence, severity, and quantity of warning/alarming statements, and the extent of any danger stated.
    2. "riesgo_explicacion": A detailed textual explanation IN SPANISH. Start your explanation by explicitly stating the assigned 'riesgo_rating' and then detail the primary reasons for this rating, referencing specific warnings, dangers, and concerns from the document. For example: 'El riesgo se considera [valor de riesgo_rating] debido a [razones principales y detalles específicos del documento].' Ensure any special characters like backslashes or quotes within this explanation are correctly escaped for JSON string format (e.g., a backslash should be '\\\\', a quote '\\"').
    3. "resumen_detallado_ia": A comprehensive and explanatory summary of the entire document, IN SPANISH. **This summary MUST be less than 1000 words.** It should focus on key findings, methodologies (if applicable), conclusions, and recommendations. Ensure any special characters like backslashes or quotes within this summary are correctly escaped for JSON string format.

    Ensure the output is ONLY a valid JSON object. Do not add any text before or after the JSON object. All string values inside the JSON must be properly JSON escaped. Example format:
    {{
      "riesgo_rating": "bajo",
      "riesgo_explicacion": "El riesgo se considera bajo porque solo se mencionaron algunas preocupaciones menores y no hay indicios de peligro inminente. Por ejemplo, una ruta de archivo podría ser C:\\\\Users\\\\temp.",
      "resumen_detallado_ia": "El documento trata sobre la \\"importancia\\" de..."
    }}

    Document Text (Spanish):
    ---
    {document_text[:1500000]} 
    ---

    JSON Output:
    """

    print("    Requesting combined (rating, explanation, summary) JSON from Gemini...")
    raw_response_text_for_debug = "No response received" # For debugging

    # --- RETRY LOGIC START ---
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Explicitly set timeout for the API call
            response = model.generate_content(prompt_combined_json, request_options={'timeout': 180})
            
            # If successful, break the loop and process response
            break 

        except (ResourceExhausted, ServiceUnavailable, InternalServerError) as e_quota:
            retry_count += 1
            wait_time = 60 * retry_count # Wait 60s, then 120s, etc.
            print(f"    API Quota Hit or Server Error ({type(e_quota).__name__}). Sleeping for {wait_time} seconds before retry {retry_count}/{max_retries}...")
            time.sleep(wait_time)
            if retry_count == max_retries:
                return f"Error: Quota Exceeded ({type(e_quota).__name__})", default_error_explanation, default_error_summary
        
        except Exception as e_api:
            # For other errors (like invalid argument), fail immediately
            print(f"    Major Error during Gemini API call: {type(e_api).__name__} - {e_api}")
            return f"API Exception: {type(e_api).__name__}", default_error_explanation, default_error_summary
    # --- RETRY LOGIC END ---

    # --- PROCESS RESPONSE (Existing Logic) ---
    try:
        if not response.parts:
            block_reason_str = "Unknown reason"
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason is not None:
                    block_reason_str = str(response.prompt_feedback.block_reason)
            return f"AI Error: No Parts ({block_reason_str})", default_error_explanation, default_error_summary

        raw_response_text_for_debug = response.text 
        text_to_parse = raw_response_text_for_debug.strip()
        json_candidate_text = None

        match_code_block = re.search(r"```json\s*(.*?)\s*```", text_to_parse, re.DOTALL)
        if match_code_block:
            json_candidate_text = match_code_block.group(1).strip()
        else:
            match_json_object = re.search(r"\{.*\}", text_to_parse, re.DOTALL)
            if match_json_object:
                json_candidate_text = match_json_object.group(0).strip()
            else:
                json_candidate_text = text_to_parse.strip()

        if not json_candidate_text:
            return "Error: No JSON Candidate", default_error_explanation, default_error_summary

        response_json = json.loads(json_candidate_text)
        categorical_riesgo = response_json.get("riesgo_rating", default_error_rating).strip().lower()
        risk_explanation = response_json.get("riesgo_explicacion", default_error_explanation).strip()
        resumen_ia = response_json.get("resumen_detallado_ia", default_error_summary).strip()

        if categorical_riesgo not in RISK_CATEGORIES and not categorical_riesgo.startswith("error:"):
            categorical_riesgo = f"Error: Invalid Category ({categorical_riesgo})"

        return categorical_riesgo, risk_explanation, resumen_ia

    except json.JSONDecodeError as e_json:
        print(f"    Error decoding JSON: {e_json}")
        return "Error: JSON Decode", f"JSON Error - {e_json}", f"JSON Error - {e_json}"
    except Exception as e_parse:
        print(f"    Error parsing response: {e_parse}")
        return "Error: Parsing", default_error_explanation, default_error_summary

# ## -------- DSPACE DATA RETRIEVAL --------

# ### Get records

def get_dspace_data(url):
    print("Connecting to DSpace repository and requesting records...")

    sickle_instance = Sickle(url, 
                             # user_agent=user_agent_string, 
                             max_retries=10, 
                             timeout=120)
    try:
        # records = sickle_instance.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True)
        records = sickle_instance.ListRecords(**{'metadataPrefix': 'oai_dc','from': DSpace_start}, ignore_deleted=True)
        mylist = list()
        for i, record in enumerate(records):
            mylist.append(record.metadata)
            if (i + 1) % 100 == 0: print(f"  Processed {i+1} records from DSpace...")
        if not mylist: print("No records found in DSpace repository."); return pd.DataFrame()
        myDF = pd.DataFrame(mylist)
        print(f"Successfully retrieved {len(myDF)} raw records from DSpace.")
        return myDF
    except Exception as e:
        print(f"Error retrieving data from DSpace: {e}"); return pd.DataFrame()

def get_dspace_data_scythe(url):
    print("Connecting to DSpace repository and requesting records...")

    scythe = Scythe(url, 
                    # user_agent=user_agent_string, 
                    max_retries=10, 
                    timeout=120)
    try:
        records = scythe.list_records(metadata_prefix='oai_dc',from_= DSpace_start, ignore_deleted=True)
        mylist = list()
        for i, record in enumerate(records):
            mylist.append(record.metadata)
            if (i + 1) % 100 == 0: print(f"  Processed {i+1} records from DSpace...")
        if not mylist: print("No records found in DSpace repository."); return pd.DataFrame()
        myDF = pd.DataFrame(mylist)
        print(f"Successfully retrieved {len(myDF)} raw records from DSpace.")
        return myDF
    except Exception as e:
        print(f"Error retrieving data from DSpace: {e}"); return pd.DataFrame()

# ### expand column with lists to new columns 

def expand_list_to_colums(df,variable_name=None):

    if variable is None: return

    # Expand the 'variable_name' column into new columns
    expanded_values = df[variable_name].apply(pd.Series)

    # You can optionally rename the new columns for clarity
    # For example, to 'Value_0', 'Value_1', etc.
    expanded_values.columns = [f'Value_{i}' for i in range(expanded_values.shape[1])]

    # expanded_values.columns = ['date_accessioned','date_available','date_issued']

    # Concatenate the new columns with the original DataFrame
    df_expanded = pd.concat([df, expanded_values], axis=1)

    # print("\nDataFrame after expanding the 'Values' column:")
    # print(df_expanded)
    return df_expanded

# ### Process records

def process_dspace_records(myDF_raw):
    if myDF_raw.empty: return pd.DataFrame()

    print("Processing DSpace records (Original Logic for DSpace fields)...")
    myDF = myDF_raw.copy()
    myDF["matter"]=False

    myDF.loc[myDF.type.isna(),"type"]=myDF.loc[myDF.type.isna(),"type"].apply(lambda x:[""])
    myDF['tipos_str'] = [', '.join(map(str, l)) for l in myDF['type']]
    s1=myDF["type"].explode(); cond = s1.str.contains('informe', case=False, na=False) 
    myDF.loc[s1[cond].index.unique(),"matter"]=True; myDF=myDF[myDF.matter].copy() 

    if myDF.empty: print("No 'informe' records after filtering."); return pd.DataFrame()

    myDF.loc[:,'resumen']=myDF.description.apply(lambda x: x[0] if isinstance(x, list) and x else None)
    myDF.loc[myDF.subject.isna(),"subject"]=myDF.loc[myDF.subject.isna(),"subject"].apply(lambda x:[""])
    myDF['topicos_str'] = [', '.join(map(str, l)) for l in myDF['subject']]

    myDF.loc[:,"publicado_str"]=myDF.date.apply(lambda x: x[2] if isinstance(x, list) and len(x) > 2 else (x[0] if isinstance(x, list) and len(x) > 0 else None))
    time.sleep(0.33)
    myDF.loc[:,"fecha_str"]=myDF.date.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
    time.sleep(0.33)

    import dateparser 
    # myDF.loc[:,"fecha_publicado"]=myDF.publicado_str.apply(lambda x: dateparser.parse(x, settings={'PREFER_DAY_OF_MONTH': 'first', "PREFER_MONTH_OF_YEAR": "first"}) if pd.notna(x) else pd.NaT)
    # myDF['fecha_publicado'] = pd.to_datetime(myDF['fecha_publicado'], errors='coerce')

    myDF['fecha_publicado'] = pd.to_datetime(myDF['fecha_str'], errors='coerce')
    myDF['fecha_publicado'] = myDF['fecha_publicado'].dt.tz_convert(CR_TZ).dt.tz_localize(None)

    myDF.loc[:,'titulo']=myDF.title.apply(lambda x: x[0] if isinstance(x, list) and x else None)
    myDF.loc[:,'title_N']=myDF.title.apply(lambda x: len(x) if isinstance(x, list) else 0) # RESTORED
    myDF['consecutivo'] = pd.NA 
    myDF.loc[myDF.title_N!=1,"consecutivo"]=myDF.loc[myDF.title_N!=1,"title"].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else pd.NA) # RESTORED
    myDF['relaciones_str'] = pd.NA 
    myDF.loc[~myDF.relation.isna(),"relaciones_str"]=myDF.loc[~myDF.relation.isna(),"relation"].apply(lambda x: x[0] if isinstance(x, list) and x else pd.NA)
    myDF.loc[~myDF.relaciones_str.isna(),"relaciones_str"]=myDF.loc[~myDF.relaciones_str.isna(),"relaciones_str"].apply(lambda x: x.replace(";","") if isinstance(x, str) else x)
    fill_consecutivo_mask = myDF.consecutivo.isna() & myDF.relaciones_str.notna()
    myDF.loc[fill_consecutivo_mask,"consecutivo"] = myDF.loc[fill_consecutivo_mask,"relaciones_str"] # RESTORED
    myDF['autores']= myDF.creator.apply(lambda x: '; '.join(map(str, x)) if isinstance(x, list) and x else None)
    myDF.loc[myDF.publisher.isna(),"publisher"]=myDF.loc[myDF.publisher.isna(),"publisher"].apply(lambda x:[""])
    myDF.loc[~myDF.publisher.isna(),'publicador']=myDF.loc[~myDF.publisher.isna(),"publisher"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    myDF.loc[:,'formato']=myDF.format.apply(lambda x: x[0] if isinstance(x, list) and x else None)
    myDF.loc[~myDF.language.isna(),'idioma']=myDF.loc[~myDF.language.isna(),"language"].apply(lambda x: x[0] if isinstance(x, list) and x else None)
    myDF.loc[:,'enlace'] = myDF.identifier.apply(lambda x: x[-1] if isinstance(x, list) and x and isinstance(x[-1], str) and x[-1].startswith('http') else None)

    # Columns to be returned by this function (original DSpace fields)
    # Renamed some for clarity (e.g., tipos_str) to avoid potential clashes later
    # if 'tipos' is used as a final name after more processing.
    original_dspace_columns = [
        'enlace', 'titulo', 'title_N', 'autores', 'fecha_publicado', 'resumen', 
        'topicos_str', 'tipos_str', 'publicador', 'formato', 'idioma', 'consecutivo', 
        'fecha_str', 'publicado_str', 'relaciones_str'
    ]
    finalDF_original_fields = pd.DataFrame()
    for col_name in original_dspace_columns: # Changed col to col_name
        if col_name in myDF.columns:
            finalDF_original_fields[col_name] = myDF[col_name]
        else: # Should ideally not happen if myDF processing is correct
            finalDF_original_fields[col_name] = 0 if col_name == 'title_N' else \
                                                (pd.NaT if col_name == 'fecha_publicado' else "")
            print(f"Warning: DSpace column '{col_name}' was missing, initialized default.")

    finalDF_original_fields.sort_values(by='fecha_publicado',ascending=False,inplace=True)
    finalDF_original_fields.reset_index(drop=True, inplace=True) 
    print(f"{len(finalDF_original_fields)} DSpace records processed for original fields.")
    return finalDF_original_fields

# ## -------- GOOGLE SHEETS INTERACTIONS --------

# ### Connect to Google Sheet

def get_google_sheet_connection(api_secret_json_str, sheet_id_or_name):
    if not api_secret_json_str: print("\nGoogle API secret is missing."); return None
    try:
        credentials_dict = json.loads(api_secret_json_str); gc = gspread.service_account_from_dict(credentials_dict)
        sh = gc.open_by_key(sheet_id_or_name); print(f"\nSuccessfully connected to Google Sheet: {sh.title}"); return sh
    except json.JSONDecodeError as e: print(f"\nError decoding Google API JSON secret: {e}."); return None
    except Exception as e: print(f"\nError connecting to Google Sheets: {e}"); return None

# ### Get master sheet

def get_master_sheet_data(sheet_connection, sheet_name): # Same as before, ensures pdf_link_direct, risk_explanation
    default_ai_cols = {'riesgo': "Not generated", 'risk_explanation': "Not generated", 'resumen_IA': "Not generated"}
    try:
        worksheet = sheet_connection.worksheet(sheet_name); print(f"Reading data from master sheet: '{sheet_name}'")
        df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0, na_filter=False, dtype=str) 
        for col_check in ['enlace', 'pdf_link_direct', 'title_N']: # Added title_N
             if col_check not in df.columns: print(f"Warning: '{col_check}' not found. Creating."); df[col_check] = "0" if col_check == 'title_N' else ""
        for col, default_val in default_ai_cols.items():
            if col not in df.columns: print(f"Column '{col}' not found. Adding with default."); df[col] = default_val
            else: df[col] = df[col].astype(str).replace('', default_val).fillna(default_val)
        if 'fecha_publicado' in df.columns: df['fecha_publicado'] = pd.to_datetime(df['fecha_publicado'], errors='coerce')
        else: print(f"Warning: 'fecha_publicado' not found."); df['fecha_publicado'] = pd.NaT
        if 'title_N' in df.columns: df['title_N'] = pd.to_numeric(df['title_N'], errors='coerce').fillna(0).astype(int) # Convert title_N to int

        print(f"Read {len(df)} records from '{sheet_name}'"); return df, worksheet
    except gspread.exceptions.WorksheetNotFound:
        print(f"Master sheet '{sheet_name}' not found. Will define structure for new one.")
        empty_cols = ['enlace', 'titulo', 'title_N', 'autores', 'fecha_publicado', 'resumen', 'topicos',
                      'tipos', 'publicador', 'formato', 'idioma', 'consecutivo', 'fecha',
                      'publicado', 'relaciones', 'pdf_link_direct',
                      'riesgo', 'risk_explanation', 'resumen_IA']
        empty_df_data = {}
        for col in empty_cols:
            if col == 'fecha_publicado': empty_df_data[col] = pd.Series(dtype='datetime64[ns]')
            elif col == 'title_N': empty_df_data[col] = pd.Series(dtype='int')
            else: empty_df_data[col] = pd.Series(dtype='str')
        return pd.DataFrame(empty_df_data), None 
    except Exception as e:
        print(f"Error reading master sheet '{sheet_name}': {e}")
        empty_cols_err = ['enlace', 'riesgo', 'risk_explanation', 'resumen_IA', 'fecha_publicado', 'pdf_link_direct', 'titulo', 'title_N']
        empty_df_data_err = {col: pd.Series(dtype='str') for col in empty_cols_err if col not in ['fecha_publicado', 'title_N']}
        empty_df_data_err['fecha_publicado'] = pd.Series(dtype='datetime64[ns]'); empty_df_data_err['title_N'] = pd.Series(dtype='int')
        return pd.DataFrame(empty_df_data_err), None

# ## -------- MAIN SCRIPT LOGIC --------

def main():
    print(f"\nScript started at:{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if not GOOGLE_API_SECRET or not GOOGLE_SHEET_ID: print("Secrets missing. Exiting."); return

    ### GET RECORDS FROM REPO DSPACE
    # raw_dspace_df = get_dspace_data(URL)
    raw_dspace_df = get_dspace_data_scythe(URL)

    # globals()['raw_dspace_df']=raw_dspace_df
    if raw_dspace_df.empty: print("No DSpace data. Exiting."); return

    ### CREATE DF WITH RECORDS
    base_df_from_dspace = process_dspace_records(raw_dspace_df)
    # globals()['base_df_from_dspace']=base_df_from_dspace

    if base_df_from_dspace.empty: print("No processable DSpace records. Exiting."); return

    ### CREATE NEW COLUMNS FOR PROCESSING
    augmented_dspace_df = base_df_from_dspace.copy()
    if 'pdf_link_direct' not in augmented_dspace_df.columns: augmented_dspace_df['pdf_link_direct'] = ""
    if 'riesgo' not in augmented_dspace_df.columns: augmented_dspace_df['riesgo'] = "Not generated"
    if 'risk_explanation' not in augmented_dspace_df.columns: augmented_dspace_df['risk_explanation'] = "Not generated"
    if 'resumen_IA' not in augmented_dspace_df.columns: augmented_dspace_df['resumen_IA'] = "Not generated"

    ### CONNECT TO GSHEET
    gs_connection = get_google_sheet_connection(GOOGLE_API_SECRET, GOOGLE_SHEET_ID)
    if not gs_connection: print("GSheet connection failed. Exiting."); return

    ### GET MASTER DF FROM GSHEET
    master_df, master_worksheet = get_master_sheet_data(gs_connection, MASTER_SHEET_NAME) 
    ALL_FINAL_COLUMNS = ['enlace', 'titulo', 'title_N', 'autores', 'fecha_publicado', 'resumen', 'topicos_str', 'tipos_str', 'publicador', 'formato', 'idioma', 'consecutivo', 'fecha_str', 'publicado_str', 'relaciones_str', 'pdf_link_direct', 'riesgo', 'risk_explanation', 'resumen_IA']

    ### CREATE MASTER IS NOT EXIST
    if master_worksheet is None: 
         print(f"Master sheet '{MASTER_SHEET_NAME}' creating attempt.")
         try:
            expected_cols_new_sheet = ALL_FINAL_COLUMNS 
            master_worksheet = gs_connection.add_worksheet(title=MASTER_SHEET_NAME, rows=1, cols=len(expected_cols_new_sheet))
            master_worksheet.update([expected_cols_new_sheet], 'A1') 
            print(f"Master sheet '{MASTER_SHEET_NAME}' created with headers: {expected_cols_new_sheet}")

            master_df = pd.DataFrame(columns=expected_cols_new_sheet) 
            if 'fecha_publicado' in master_df.columns: master_df['fecha_publicado'] = pd.to_datetime(master_df['fecha_publicado'], errors='coerce')
            if 'title_N' in master_df.columns: master_df['title_N'] = pd.to_numeric(master_df['title_N'], errors='coerce').fillna(0).astype(int)
            for col_init in ALL_FINAL_COLUMNS:
                 init_val = "Not generated" if col_init in ['riesgo', 'risk_explanation', 'resumen_IA'] else (0 if col_init == 'title_N' else (pd.NaT if col_init == 'fecha_publicado' else ""))
                 if col_init not in master_df.columns: master_df[col_init] = init_val 
                 else: 
                    if col_init == 'fecha_publicado': master_df[col_init] = pd.to_datetime(master_df[col_init], errors='coerce').fillna(pd.NaT)
                    elif col_init == 'title_N': master_df[col_init] = pd.to_numeric(master_df[col_init], errors='coerce').fillna(0).astype(int)
                    else: master_df[col_init] = master_df[col_init].astype(str).fillna(init_val)
         except Exception as e: print(f"Could not create master sheet: {e}. Exiting."); return

    ### BUILD PDF LINKS
    print("\nIdentifying records for direct PDF link extraction...")

    for col_check_master in ['enlace', 'pdf_link_direct']:
        if col_check_master not in master_df.columns: master_df[col_check_master] = ""
        else: master_df[col_check_master] = master_df[col_check_master].astype(str).fillna("")

    existing_enlaces_in_master = set(master_df['enlace'].loc[master_df['enlace'] != ''])
    new_dspace_records_needing_check = augmented_dspace_df[~augmented_dspace_df['enlace'].astype(str).isin(existing_enlaces_in_master)].copy()
    enlaces_from_new_dspace_records = set(new_dspace_records_needing_check['enlace'].dropna())
    print(f"Found {len(enlaces_from_new_dspace_records)} 'enlace' URLs from new DSpace records for PDF link check.")
    enlaces_from_master_needing_pdf_link = set(master_df.loc[(master_df['pdf_link_direct'] == '') & (master_df['enlace'] != ''), 'enlace'].dropna())
    print(f"Found {len(enlaces_from_master_needing_pdf_link)} 'enlace' URLs in Master for PDF link check.")
    unique_enlaces_to_fetch_pdf_for = enlaces_from_new_dspace_records.union(enlaces_from_master_needing_pdf_link)
    print(f"Total unique landing pages to process for PDF links: {len(unique_enlaces_to_fetch_pdf_for)}")
    enlace_to_direct_pdf_map = {}

    if unique_enlaces_to_fetch_pdf_for:
        print("Starting Selenium/BS4 for direct PDF links...")
        processed_landing_count = 0
        for i, landing_url in enumerate(list(unique_enlaces_to_fetch_pdf_for)): 
            print(f"  Processing landing page {i+1}/{len(unique_enlaces_to_fetch_pdf_for)}: {landing_url}")
            if pd.notna(landing_url) and isinstance(landing_url, str) and landing_url.startswith('http'):
                try:
                    direct_pdf_url = get_pdf_url(landing_url) 
                    enlace_to_direct_pdf_map[landing_url] = direct_pdf_url if direct_pdf_url else "" 
                    if direct_pdf_url: print(f"    Mapped '{landing_url}' to direct PDF: '{direct_pdf_url}'")
                    else: print(f"    Could not extract PDF URL for '{landing_url}'.")
                    processed_landing_count +=1
                    if processed_landing_count > 0 and processed_landing_count % 5 == 0: print(f"    Pausing after {processed_landing_count} pages..."); time.sleep(3)
                except Exception as e_sel_main: print(f"    Error get_pdf_url for '{landing_url}': {e_sel_main}"); enlace_to_direct_pdf_map[landing_url] = ""
            else: enlace_to_direct_pdf_map[landing_url] = "" 
        print("Finished Selenium/BS4 processing.")

    if 'pdf_link_direct' not in augmented_dspace_df.columns: augmented_dspace_df['pdf_link_direct'] = ""
    augmented_dspace_df['pdf_link_direct'] = augmented_dspace_df['enlace'].map(enlace_to_direct_pdf_map).fillna(augmented_dspace_df['pdf_link_direct'])
    if not master_df.empty and enlace_to_direct_pdf_map:
        update_mask_master = (master_df['pdf_link_direct'] == '') & (master_df['enlace'].isin(enlace_to_direct_pdf_map.keys()))
        mapped_values_master = master_df.loc[update_mask_master, 'enlace'].map(enlace_to_direct_pdf_map)
        if not mapped_values_master.empty: master_df.loc[update_mask_master, 'pdf_link_direct'] = mapped_values_master.fillna(""); print("Updated 'pdf_link_direct' in master_df.")

    ### GET NEW RECORDS TO APPEND MASTER
    if master_df.empty or not existing_enlaces_in_master: 
        new_records_df_for_append = augmented_dspace_df.copy()
    else: 
        new_records_df_for_append = augmented_dspace_df[~augmented_dspace_df['enlace'].astype(str).isin(existing_enlaces_in_master)].copy()
    print(f"Final count of new records for appending: {len(new_records_df_for_append)}")

    ### APPEND NEW RECORDS
    if not new_records_df_for_append.empty:
        print(f"Processing {len(new_records_df_for_append)} new records for sheets...")
        df_to_write_dated_sheet = new_records_df_for_append.copy()

        for idx, row_new in df_to_write_dated_sheet.iterrows():
            pub_date = row_new.get('fecha_publicado')
            if pd.notna(pub_date) and hasattr(pub_date, 'year') and (pub_date < GEMINI_PROCESSING_CUTOFF_DATE):
                df_to_write_dated_sheet.loc[idx, ['riesgo', 'risk_explanation', 'resumen_IA']] = "Not processed (Old)"

        date_str = datetime.today().strftime('%Y-%m-%d'); new_sheet_title = f'New_{date_str}'

        try:
            cols_for_dated_sheet = ALL_FINAL_COLUMNS 
            final_df_for_dated_sheet = df_to_write_dated_sheet.reindex(columns=cols_for_dated_sheet).copy()
            try: 
                new_records_worksheet = gs_connection.worksheet(new_sheet_title); new_records_worksheet.clear(); print(f"Cleared dated sheet: '{new_sheet_title}'.")
            except gspread.exceptions.WorksheetNotFound: 
                new_records_worksheet = gs_connection.add_worksheet(title=new_sheet_title, rows=max(1, len(final_df_for_dated_sheet) + 1), cols=len(cols_for_dated_sheet))

            for col_dt in final_df_for_dated_sheet.select_dtypes(include=['datetime64[ns]']).columns:
                final_df_for_dated_sheet[col_dt] = \
                final_df_for_dated_sheet[col_dt].apply(lambda x: x.isoformat() if pd.notnull(x) and hasattr(x, 'isoformat') else "")

            final_df_for_dated_sheet = final_df_for_dated_sheet.fillna('')
            set_with_dataframe(new_records_worksheet, final_df_for_dated_sheet, include_index=False, resize=True)
            print(f"Saved {len(final_df_for_dated_sheet)} new records to dated sheet: '{new_sheet_title}'")
            df_to_append_master_gs = df_to_write_dated_sheet.copy()
            master_df_before_append_len = len(master_df)
            try: 
                master_headers = master_worksheet.row_values(1) if master_worksheet.row_count > 0 else ALL_FINAL_COLUMNS
                df_prep_gs_final = df_to_append_master_gs.reindex(columns=master_headers).copy()

                for col_dt_gs in df_prep_gs_final.select_dtypes(include=['datetime64[ns]']).columns:
                    df_prep_gs_final[col_dt_gs] = df_prep_gs_final[col_dt_gs].apply(lambda x: x.isoformat() if pd.notnull(x) and hasattr(x, 'isoformat') else "")

                df_prep_gs_final = df_prep_gs_final.fillna('')
                df_for_gs_append_final_vals = df_prep_gs_final.values.tolist()

                if df_for_gs_append_final_vals: 
                    print(f"Appending {len(df_for_gs_append_final_vals)} rows to master."); 
                    master_worksheet.append_rows(df_for_gs_append_final_vals, value_input_option='USER_ENTERED'); 
                    print("Appended to master.")
                else: print("No data to append to master.")

                temp_master_df = master_df.copy(); temp_df_to_append = df_to_write_dated_sheet.copy()
                all_cols_concat = list(set(temp_master_df.columns) | set(temp_df_to_append.columns)); 
                if not all_cols_concat: all_cols_concat = ALL_FINAL_COLUMNS
                temp_master_df = temp_master_df.reindex(columns=all_cols_concat); 
                temp_df_to_append = temp_df_to_append.reindex(columns=all_cols_concat)
                master_df = pd.concat([temp_master_df, temp_df_to_append], ignore_index=True)

                if 'fecha_publicado' in master_df.columns: 
                    master_df['fecha_publicado'] = pd.to_datetime(master_df['fecha_publicado'], errors='coerce')

                if 'title_N' in master_df.columns: 
                    master_df['title_N'] = pd.to_numeric(master_df['title_N'], errors='coerce').fillna(0).astype(int)

                for col_ai_c in ALL_FINAL_COLUMNS:
                    init_val_c = "Not generated" if col_ai_c in ['riesgo', 'risk_explanation', 'resumen_IA'] else (0 if col_ai_c == 'title_N' else (pd.NaT if col_ai_c == 'fecha_publicado' else ""))
                    if col_ai_c not in master_df.columns: 
                        master_df[col_ai_c] = init_val_c
                    else: 
                        if col_ai_c == 'fecha_publicado': master_df[col_ai_c] = pd.to_datetime(master_df[col_ai_c], errors='coerce').fillna(pd.NaT)
                        elif col_ai_c == 'title_N': master_df[col_ai_c] = pd.to_numeric(master_df[col_ai_c], errors='coerce').fillna(0).astype(int)
                        elif col_ai_c in ['riesgo', 'risk_explanation', 'resumen_IA']: master_df[col_ai_c] = master_df[col_ai_c].astype(str).fillna("Not generated")
                        else: master_df[col_ai_c] = master_df[col_ai_c].astype(str).fillna("")
                print(f"In-memory master_df updated. Length: {len(master_df)} (was {master_df_before_append_len})")
            except Exception as e_append: print(f"ERROR during append/master_df update: {e_append}")
        except Exception as e_dated: print(f"Error saving to dated sheet: {e_dated}")
    else: print("No new records to add.")

    ### PROCESS WITH GEMINI
    if not GEMINI_API_KEY or model is None: print("Gemini API not configured. Skipping AI.")
    else:
        print("\nStarting AI Analysis..."); master_df_updated = master_df.copy()

        for col_ai_upd in ['riesgo', 'risk_explanation', 'resumen_IA']:
            if col_ai_upd not in master_df_updated.columns: master_df_updated[col_ai_upd] = "Not generated"
            else: master_df_updated[col_ai_upd] = master_df_updated[col_ai_upd].astype(str).replace('', "Not generated").fillna("Not generated")

        if 'pdf_link_direct' not in master_df_updated.columns: master_df_updated['pdf_link_direct'] = "" 
        else: master_df_updated['pdf_link_direct'] = master_df_updated['pdf_link_direct'].astype(str).fillna("")

        processed_for_ai_count = 0
        records_to_process_mask = (master_df_updated['riesgo'].astype(str).str.strip().str.lower() == "not generated") | (master_df_updated['risk_explanation'].astype(str).str.strip().str.lower() == "not generated") | (master_df_updated['resumen_IA'].astype(str).str.strip().str.lower() == "not generated")
        records_to_process_indices = master_df_updated[records_to_process_mask].index
        print(f"Found {len(records_to_process_indices)} records for AI. Checking constraints...")

        for index_loop in records_to_process_indices:
            row_loop = master_df_updated.loc[index_loop]; doc_title_loop = str(row_loop.get('titulo', 'N/A')); actual_pdf_to_process = str(row_loop.get('pdf_link_direct', '')).strip(); publish_date_loop = row_loop.get('fecha_publicado')

            if pd.isna(publish_date_loop): master_df_updated.loc[index_loop, ['riesgo', 'risk_explanation', 'resumen_IA']] = "Not processed (Missing Date)"; continue 

            if not (isinstance(publish_date_loop, pd.Timestamp) or isinstance(publish_date_loop, datetime)):
                 try: publish_date_loop = pd.to_datetime(publish_date_loop)
                 except: pass 

            if not (isinstance(publish_date_loop, pd.Timestamp) or isinstance(publish_date_loop, datetime)) or pd.isna(publish_date_loop): master_df_updated.loc[index_loop, ['riesgo', 'risk_explanation', 'resumen_IA']] = "Not processed (Bad Date)"; continue

            if publish_date_loop < GEMINI_PROCESSING_CUTOFF_DATE: master_df_updated.loc[index_loop, ['riesgo', 'risk_explanation', 'resumen_IA']] = "Not processed (Old)"; continue 
            print(f"\nProcessing AI for Idx {index_loop} (T: {doc_title_loop}, Date: {publish_date_loop.strftime('%Y-%m-%d')}) Direct PDF: '{actual_pdf_to_process}'")

            if not actual_pdf_to_process or not actual_pdf_to_process.lower().startswith('http'):
                print(f"  Invalid direct PDF link ('{actual_pdf_to_process}')."); 
                for fld_ai in ['riesgo', 'risk_explanation', 'resumen_IA']: 
                    if str(master_df_updated.loc[index_loop, fld_ai]).strip().lower() == "not generated": master_df_updated.loc[index_loop, fld_ai] = "Invalid Direct PDF Link"
                continue

            ### EXTRACT TEXT FROM PDF with FITZ
            document_text = None
            print(f" Attempting text extraction with PyMuPDF (fitz) for '{actual_pdf_to_process}'...")
            document_text_raw = extract_text_from_pdf_fitz(actual_pdf_to_process)

            ### EXTRACT TEXT FROM PDF with OCRMYPDF
            if not document_text_raw:
                print(f" Fitz extraction failed or yielded no text. Attempting OCRmyPDF for '{actual_pdf_to_process}'...")
                document_text_raw = extract_text_from_pdf_ocred(actual_pdf_to_process, lang_code='spa') # Your existing call

            ### EXTRACT TEXT FROM PDF with PYTESSERACT
            if not document_text_raw:
                print(f" OCRmyPDF also failed or yielded no text. Attempting Pytesseract OCR fallback for '{actual_pdf_to_process}'...")
                try:
                    pytesseract.get_tesseract_version() # Check if Tesseract is callable
                    document_text_raw = extract_text_from_pdf_ocr(actual_pdf_to_process, lang_code='spa')
                    if document_text_raw:
                        print(" Pytesseract OCR fallback extracted text.")
                    else:
                        print(" Pytesseract OCR fallback also failed or yielded no text.")
                except Exception as e_tess:
                    print(f" Pytesseract OCR fallback skipped: Tesseract not working or error: {e_tess}")

            ### PREPPING AND PASSING TEXT TO GEMINI
            if document_text_raw:
                print(f" Raw extracted text length: {len(document_text_raw)}")
                ### CLEAN TEXT
                document_text_cleaned = clean_text_for_llm(document_text_raw) # <--- APPLY CLEANING
                print(f" Cleaned text length: {len(document_text_cleaned)}. Sending to Gemini...")
                ### APPLY LLM 
                riesgo_cat_val, riesgo_expl_val, resumen_ia_val = get_gemini_analysis(document_text_cleaned)
                if str(master_df_updated.loc[index_loop, 'riesgo']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'riesgo'] = riesgo_cat_val
                if str(master_df_updated.loc[index_loop, 'risk_explanation']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'risk_explanation'] = riesgo_expl_val
                if str(master_df_updated.loc[index_loop, 'resumen_IA']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'resumen_IA'] = resumen_ia_val
                print(f"    R: {master_df_updated.loc[index_loop, 'riesgo']}, Expl: {master_df_updated.loc[index_loop, 'risk_explanation'][:50]}..., SumIA: {master_df_updated.loc[index_loop, 'resumen_IA'][:50]}...")
                processed_for_ai_count +=1
                time.sleep(5)
            else:
                ### IN CASE OF NOT HAVING TEXT FROM PDF
                print(f"  Failed text extraction from '{actual_pdf_to_process}'."); 
                for fld_ai_fail in ['riesgo', 'risk_explanation', 'resumen_IA']: 
                    if str(master_df_updated.loc[index_loop, fld_ai_fail]).strip().lower() == "not generated": master_df_updated.loc[index_loop, fld_ai_fail] = "PDF Text Extraction Failed (All Methods)"
            if processed_for_ai_count > 0 and processed_for_ai_count % 3 == 0 : print("  Pausing (3 docs)..."); time.sleep(10)

        ### UPDATE MASTER AFTER GETTING GEMINI RESULTS
        if processed_for_ai_count > 0 or len(records_to_process_indices) > 0: 
            print(f"\nAI processing finished. {processed_for_ai_count} to Gemini. Updating master..."); 
            try:
                master_headers_final = master_worksheet.row_values(1) if master_worksheet.row_count > 0 else ALL_FINAL_COLUMNS
                final_df_to_upload = master_df_updated.copy().reindex(columns=master_headers_final)

                for col_final_dt_upd in final_df_to_upload.select_dtypes(include=['datetime64[ns]']).columns: 
                    final_df_to_upload[col_final_dt_upd] = final_df_to_upload[col_final_dt_upd].apply(lambda x: x.isoformat() if pd.notnull(x) and hasattr(x, 'isoformat') else "")

                final_df_to_upload = final_df_to_upload.fillna('') 
                set_with_dataframe(master_worksheet, final_df_to_upload, include_index=False, resize=True); print("Master sheet updated.")
            except Exception as e_master_upd: print(f"Error updating master sheet: {e_master_upd}")
        else: print("No records required AI updates.")
    print("\nScript finished at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

# # Execute

if __name__ == '__main__':
    main()