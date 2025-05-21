# %% [code]
# %% [code]
# coding: utf-8

# Instalar librerías no estándares
import os
os.system('pip install -U pip')
os.system('pip install sickle')
os.system('pip install dateparser')
os.system('pip install python-dateutil')
os.system('pip install gspread gspread_dataframe google-api-python-client')
os.system('pip install google-generativeai requests pdfplumber')
os.system('pip install pymupdf')
os.system('pip install selenium beautifulsoup4')
os.system('pip install pytesseract Pillow') # For OCR
# In Kaggle, ensure geckodriver:
# os.subprocess(["apt-get", "update"],"apt-get update")
# os.subprocess(["apt-get", "install", "-y", "firefox", "geckodriver"],"apt-get install -y firefox geckodriver")
# os.subprocess(["apt-get","install","-y","tesseract-ocr","tesseract-ocr-spa"],"apt-get install -y tesseract-ocr tesseract-ocr-spa")

! apt-get update -y && apt-get install -y tesseract-ocr tesseract-ocr-spa && apt-get install -y tesseract-ocr tesseract-ocr-spa

print("\nScript to request records from LANAMME's repository and update our data with AI analysis")

import datetime as dt
import pandas as pd
import numpy as np
import re
import time 
from datetime import datetime
import json 
from io import BytesIO
import fitz


from sickle import Sickle
import gspread
try:
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
except ImportError: 
    os.system('pip install gspread_dataframe')
    from gspread_dataframe import set_with_dataframe, get_as_dataframe
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Interactive', 'Batch'):
    from kaggle_secrets import UserSecretsClient
else: 
    os.system('pip install python-dotenv') 
    from dotenv import load_dotenv, find_dotenv
    if find_dotenv(): print("Found .env file, loading."); _ = load_dotenv(find_dotenv()) 
    else: print(".env file not found for local dev.")

import google.generativeai as genai
import requests
import pdfplumber 
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as FirefoxService 
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin
import pytesseract # For OCR
from PIL import Image # For OCR with Tesseract


URL = 'https://www.lanamme.ucr.ac.cr/oai/request?'
MASTER_SHEET_NAME = 'Master' 
RISK_CATEGORIES = ["ninguno", "bajo", "medio", "alto", "critico"] 

# NEW: Define the cutoff date for Gemini AI processing
# Records published strictly BEFORE this date will skip Gemini analysis
# datetime(YEAR, MONTH, DAY)
GEMINI_PROCESSING_CUTOFF_DATE = datetime(2024, 1, 1) 



# --- SECRET LOADING ---
if os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Interactive', 'Batch'):
    print("Kaggle: Loading secrets."); user_secrets = UserSecretsClient()
    try: GEMINI_API_KEY = user_secrets.get_secret("GEMINI_API_KEY"); GOOGLE_API_SECRET = user_secrets.get_secret("Google_API"); GOOGLE_SHEET_ID = user_secrets.get_secret("NewSheetID"); print("Kaggle secrets loaded.")
    except Exception as e: print(f"Error Kaggle Secrets: {e}"); GEMINI_API_KEY, GOOGLE_API_SECRET, GOOGLE_SHEET_ID = None, None, None
else: 
    print("Local: Loading secrets from env.");
    try: GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]; GOOGLE_API_SECRET = os.environ["Google_API"]; GOOGLE_SHEET_ID = os.environ["NewSheetID"]; print("Local env secrets loaded.")
    except KeyError as e: print(f"Error Local Env Var: {e} not set."); GEMINI_API_KEY, GOOGLE_API_SECRET, GOOGLE_SHEET_ID = None, None, None
    except Exception as e: print(f"Error Local Secrets: {e}"); GEMINI_API_KEY, GOOGLE_API_SECRET, GOOGLE_SHEET_ID = None, None, None

# --- GEMINI MODEL CONFIG ---
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY); GEMINI_MODEL_NAME = 'gemini-2.5-flash-preview-04-17' 
    try: generation_config = genai.types.GenerationConfig(response_mime_type="application/json"); model = genai.GenerativeModel(GEMINI_MODEL_NAME, generation_config=generation_config); print(f"Gemini model '{GEMINI_MODEL_NAME}' initialized for JSON.")
    except Exception as e: print(f"Error initializing Gemini: {e}"); model = None
else: model = None; print("Gemini API Key not found/model init failed. AI processing skipped.")

# --- HELPER FUNCTIONS ---

def get_pdf_url(url):

    options = FirefoxOptions()
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

def extract_text_from_pdf_fitz(pdf_url_direct):
    """
    Fetches a PDF document from a given DIRECT PDF URL and extracts its text content
    using PyMuPDF (fitz).

    Args:
        pdf_url_direct (str): The URL of the PDF document to process.

    Returns:
        str or None: The extracted text content from the PDF as a single string,
                     or None if any error occurs.
    """
    if not pdf_url_direct or not isinstance(pdf_url_direct, str) or not pdf_url_direct.lower().startswith('http'):
        print(f"  Fitz: Error - Invalid URL format for PDF extraction: '{pdf_url_direct}'")
        return None
    
    # Optional: A warning if it doesn't end with .pdf, though PyMuPDF might handle it if Content-Type is correct
    if not pdf_url_direct.lower().endswith('.pdf'):
         print(f"  Fitz: Warning - Direct PDF URL does not end with .pdf: '{pdf_url_direct}'. Attempting download.")

    try:
        print(f"  Fitz: Attempting to download direct PDF from: {pdf_url_direct}")
        response = requests.get(pdf_url_direct, timeout=45, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()  
        
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"  Fitz: ERROR - Content-Type is '{content_type}', not 'application/pdf'. URL '{pdf_url_direct}' may not be a direct PDF link.")
            return None 
        
        print(f"  Fitz: Direct PDF downloaded (status {response.status_code}). Size: {len(response.content)} bytes.")
        
        pdf_bytes = response.content
        text_content = ""
        
        # Open PDF from bytes using PyMuPDF
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            print(f"  Fitz: Extracting text from {len(doc)} page(s)...")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                page_text = page.get_text("text") # Get plain text; other options: "html", "xml", "xhtml", "json"
                if page_text:
                    text_content += page_text + "\n"
            doc.close()
            if text_content:
                print(f"  Fitz: Text extraction complete.")
            else:
                print(f"  Fitz: No text could be extracted (PDF might be image-based or empty of text).")
        except Exception as e_fitz:
            # This catches errors if fitz.open() or page.get_text() fails
            print(f"  Fitz: Error opening or parsing PDF with PyMuPDF from '{pdf_url_direct}': {e_fitz}")
            return None # Return None if PyMuPDF fails
        
        return text_content.strip() if text_content.strip() else None # Return None if only whitespace or empty

    except requests.exceptions.Timeout:
        print(f"Fitz: Error downloading PDF from '{pdf_url_direct}': Request timed out.")
    except requests.exceptions.RequestException as e_req:
        print(f"Fitz: Error downloading PDF from '{pdf_url_direct}': {e_req}")
    except Exception as e_generic:
        print(f"Fitz: Generic error during PDF processing for '{pdf_url_direct}': {e_generic}")
    return None


def extract_text_from_pdf_ocr(pdf_url_direct, lang_code='spa'):
    """
    Fetches a PDF from a URL, renders its pages as images, and extracts text using OCR.
    This is a fallback method and can be slower and more resource-intensive.

    Args:
        pdf_url_direct (str): The direct URL of the PDF document.
        lang_code (str): The language code for Tesseract OCR (e.g., 'spa' for Spanish).

    Returns:
        str or None: The extracted text content from OCR, or None if errors occur.
    """
    if not pdf_url_direct or not isinstance(pdf_url_direct, str) or not pdf_url_direct.lower().startswith('http'):
        print(f"  OCR: Error - Invalid URL for PDF: '{pdf_url_direct}'")
        return None

    try:
        print(f"  OCR: Attempting download for OCR: {pdf_url_direct}")
        response = requests.get(pdf_url_direct, timeout=45, headers={'User-Agent': 'Mozilla/5.0'})
        response.raise_for_status()
        
        content_type = response.headers.get('content-type', '').lower()
        if 'application/pdf' not in content_type:
            print(f"  OCR: ERROR - Content-Type is '{content_type}', not PDF. URL '{pdf_url_direct}'")
            return None
            
        pdf_bytes = response.content
        print(f"  OCR: PDF downloaded for OCR. Size: {len(pdf_bytes)} bytes.")
        
        all_ocr_text = ""
        doc = None
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            print(f"  OCR: Processing {len(doc)} pages with Tesseract (lang: {lang_code})...")
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                # Render page to a pixmap (image). Higher DPI can improve OCR but is slower.
                pix = page.get_pixmap(dpi=300) 
                img_bytes = pix.tobytes("png") # Convert pixmap to PNG image bytes
                
                try:
                    pil_image = Image.open(BytesIO(img_bytes))
                    # Use Tesseract to extract text from the image
                    page_ocr_text = pytesseract.image_to_string(pil_image, lang=lang_code)
                    if page_ocr_text:
                        all_ocr_text += page_ocr_text + "\n\n" # Add extra newline for page separation
                    print(f"    OCR: Page {page_num + 1} processed.")
                except Exception as e_ocr_page:
                    print(f"    OCR: Error processing page {page_num + 1} with Tesseract: {e_ocr_page}")
            if all_ocr_text.strip():
                print("  OCR: Text extraction via OCR complete.")
            else:
                print("  OCR: No text extracted via OCR (document might be truly empty or OCR failed on all pages).")
        except Exception as e_ocr_fitz:
            print(f"  OCR: Error opening PDF with PyMuPDF for OCR: {e_ocr_fitz}")
            return None # Cannot proceed if PDF can't be opened to render pages
        finally:
            if doc:
                doc.close()
        
        return all_ocr_text.strip() if all_ocr_text.strip() else None

    except requests.exceptions.Timeout:
        print(f"OCR: Timeout downloading PDF for OCR: '{pdf_url_direct}'")
    except requests.exceptions.RequestException as e_req_ocr:
        print(f"OCR: Request error downloading PDF for OCR: {e_req_ocr}")
    except Exception as e_generic_ocr:
        print(f"OCR: Generic error during PDF download/setup for OCR: {e_generic_ocr}")
    return None

# Ensure RISK_CATEGORIES is defined globally or accessible in this function's scope
# RISK_CATEGORIES = ["ninguno", "bajo", "medio", "alto", "critico"] 

def get_gemini_analysis(document_text):
    # Initialize default error values at the very beginning
    default_error_rating = "Error: AI Analysis default"
    default_error_explanation = "Error: AI Explanation default"
    default_error_summary = "Error: AI Summary default"

    # These variables will hold the final values to be returned
    final_rating = default_error_rating
    final_explanation = default_error_explanation
    final_summary = default_error_summary

    if not model or not document_text:
        print("   Gemini model/text unavailable.")
        return final_rating, final_explanation, final_summary

    # Refined prompt to emphasize stricter JSON and escaping
    prompt_combined_json = f"""
    Analyze the following document text, which is in Spanish.
    Based on your analysis, generate a JSON object with the following three keys:
    1. "riesgo_rating": A single string value representing the overall risk level. Choose EXCLUSIVELY from this list: {RISK_CATEGORIES}. This rating should be based on the presence, severity, and quantity of warning/alarming statements, and the extent of any danger stated.
    2. "riesgo_explicacion": A detailed textual explanation IN SPANISH. Start your explanation by explicitly stating the assigned 'riesgo_rating' and then detail the primary reasons for this rating, referencing specific warnings, dangers, and concerns from the document. For example: 'El riesgo se considera [valor de riesgo_rating] debido a [razones principales y detalles específicos del documento].' Ensure any special characters like backslashes or quotes within this explanation are correctly escaped for JSON string format (e.g., a backslash should be '\\\\', a quote '\"').
    3. "resumen_detallado_ia": A comprehensive and explanatory summary of the entire document, IN SPANISH. This summary should focus on key findings, methodologies (if applicable), conclusions, and recommendations. Ensure any special characters like backslashes or quotes within this summary are correctly escaped for JSON string format.

    Ensure the output is ONLY a valid JSON object. Do not add any text before or after the JSON object. All string values inside the JSON must be properly JSON escaped. Example format:
    {{
      "riesgo_rating": "bajo",
      "riesgo_explicacion": "El riesgo se considera bajo porque solo se mencionaron algunas preocupaciones menores y no hay indicios de peligro inminente. Por ejemplo, una ruta de archivo podría ser C:\\\\Users\\\\temp.",
      "resumen_detallado_ia": "El documento trata sobre la \\"importancia\\" de..."
    }}

    Document Text (Spanish):
    ---
    {document_text[:150000]} 
    ---

    JSON Output:
    """

    print("   Requesting JSON (rating, explanation, summary) from Gemini...")
    
    raw_text_from_gemini_for_debug = "No response received" 
    json_candidate_text_for_debug = "No JSON candidate formed"
    
    try:
        response = model.generate_content(prompt_combined_json, request_options={'timeout': 180}) 

        if not response.parts: 
            block_reason_str = "Unknown reason" 
            if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                if hasattr(response.prompt_feedback, 'block_reason') and response.prompt_feedback.block_reason is not None:
                    block_reason_str = str(response.prompt_feedback.block_reason)
                elif hasattr(response.prompt_feedback, 'safety_ratings'): 
                    print(f"     Safety ratings: {response.prompt_feedback.safety_ratings}")
                    for rating_info in response.prompt_feedback.safety_ratings: 
                        if rating_info.probability not in [genai.types.HarmProbability.NEGLIGIBLE, genai.types.HarmProbability.LOW]:
                             block_reason_str = f"Safety block - Category: {rating_info.category}, Probability: {rating_info.probability.name}"
                             break 
            print(f"   Gemini API Warning: No content parts in response. Effective Block reason: {block_reason_str}.")
            final_rating = f"AI Error: No Parts ({block_reason_str})"
            return final_rating, final_explanation, final_summary

        raw_text_from_gemini_for_debug = response.text 
        
        try:
            text_to_parse = raw_text_from_gemini_for_debug.strip()
            
            # Attempt to find the JSON block more robustly
            # Remove markdown backticks first if they exist
            if text_to_parse.startswith("```json"):
                text_to_parse = text_to_parse[len("```json"):]
            if text_to_parse.endswith("```"):
                text_to_parse = text_to_parse[:-len("```")]
            text_to_parse = text_to_parse.strip() # Strip again after removing backticks

            # Now, find the first '{' and last '}'
            first_brace = text_to_parse.find('{')
            last_brace = text_to_parse.rfind('}')

            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_candidate_text = text_to_parse[first_brace : last_brace+1]
                print(f"   Attempting JSON parse (extracted from braces): {json_candidate_text[:100]}...")
            else: 
                # If no clear braces, use the cleaned text (less likely to be correct if not wrapped in braces)
                json_candidate_text = text_to_parse 
                print(f"   No clear JSON braces found after stripping markdown, attempting parse on: {json_candidate_text[:100]}...")
            
            json_candidate_text_for_debug = json_candidate_text 

            try:
                response_json = json.loads(json_candidate_text) 
            except json.JSONDecodeError as e_strict:
                print(f"   JSONDecodeError (standard strict parsing): {e_strict}. Trying with strict=False.")
                try:
                    # This is unlikely to fix "Invalid \uXXXX escape" or "Extra data" if the structure is wrong
                    response_json = json.loads(json_candidate_text, strict=False) 
                    print("   Successfully parsed with strict=False.")
                except json.JSONDecodeError as e_non_strict:
                    print(f"   JSONDecodeError (strict=False also failed): {e_non_strict}.")
                    raise e_strict # Re-raise the original, more common error for outer handling

            final_rating = response_json.get("riesgo_rating", default_error_rating).strip().lower()
            final_explanation = response_json.get("riesgo_explicacion", default_error_explanation).strip()
            final_summary = response_json.get("resumen_detallado_ia", default_error_summary).strip()

            if final_rating not in RISK_CATEGORIES and not final_rating.startswith("Error:") and not final_rating.startswith("AI Error:") and not final_rating.startswith("API Exception:"):
                print(f"   Warning: Gemini returned an invalid risk category: '{final_rating}'.")
                final_rating = f"Error: Invalid Category ({final_rating})"
            
        except json.JSONDecodeError as e_json:
            print(f"   FATAL Error decoding JSON from Gemini: {e_json}.")
            print(f"   --- Start of problematic text that caused JSON error (length: {len(json_candidate_text_for_debug)}) ---")
            # Print the exact string that caused the error, replacing non-printable characters for logging
            printable_debug_text = ''.join(c if c.isprintable() or c.isspace() else f'\\x{ord(c):02x}' for c in json_candidate_text_for_debug)
            print(printable_debug_text) 
            print(f"   --- End of problematic text ---")
            final_rating = "Error: JSON Decode"
            final_explanation = f"JSON Error - {e_json}" 
            final_summary = f"JSON Error - {e_json}"
        except AttributeError as e_attr: 
            print(f"   Error accessing Gemini response attributes (e.g., .text is None): {e_attr}")
            final_rating = "Error: Response Attribute"
        except Exception as e_parse: 
            print(f"   Unexpected error during Gemini response parsing: {e_parse}")
            final_rating = "Error: Parse Exception"

    except Exception as e_api: 
        print(f"   Major Error during Gemini API call or critical response issue: {type(e_api).__name__} - {e_api}")
        final_rating = f"API Exception: {type(e_api).__name__}"
        
    return final_rating, final_explanation, final_summary
    


def get_dspace_data(url): # Same as before
    print("Connecting to DSpace repository..."); sickle_instance = Sickle(url, max_retries=5, timeout=60) 
    try:
        records = sickle_instance.ListRecords(metadataPrefix='oai_dc', ignore_deleted=True); mylist = list()
        for i, record in enumerate(records): mylist.append(record.metadata); 
        if (i + 1) % 100 == 0: print(f"  Processed {i+1} DSpace records...")
        if not mylist: print("No records from DSpace."); return pd.DataFrame()
        myDF = pd.DataFrame(mylist); print(f"Retrieved {len(myDF)} raw DSpace records."); return myDF
    except Exception as e: print(f"Error DSpace: {e}"); return pd.DataFrame()

# MODIFIED process_dspace_records
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
    import dateparser 
    myDF.loc[:,"fecha_publicado"]=myDF.publicado_str.apply(lambda x: dateparser.parse(x, settings={'PREFER_DAY_OF_MONTH': 'first', "PREFER_MONTH_OF_YEAR": "first"}) if pd.notna(x) else pd.NaT)
    myDF['fecha_publicado'] = pd.to_datetime(myDF['fecha_publicado'], errors='coerce')
    myDF.loc[:,"fecha_str"]=myDF.date.apply(lambda x: x[0] if isinstance(x, list) and x else None)
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

def get_google_sheet_connection(api_secret_json_str, sheet_id_or_name): # Same as before
    if not api_secret_json_str: print("Google API secret is missing."); return None
    try: credentials_dict = json.loads(api_secret_json_str); gc = gspread.service_account_from_dict(credentials_dict); sh = gc.open_by_key(sheet_id_or_name); print(f"Connected to GSheet: {sh.title}"); return sh
    except Exception as e: print(f"Error connecting GSheet: {e}"); return None

def get_master_sheet_data(sheet_connection, sheet_name): # Ensures all new columns too
    default_ai_cols = {'riesgo': "Not generated", 'risk_explanation': "Not generated", 'resumen_IA': "Not generated"}
    # ALL columns expected in the master sheet for robust handling
    all_expected_master_cols = [
        'enlace', 'titulo', 'title_N', 'autores', 'fecha_publicado', 'resumen', 
        'topicos_str', 'tipos_str', 'publicador', 'formato', 'idioma', 'consecutivo', 
        'fecha_str', 'publicado_str', 'relaciones_str', 
        'pdf_link_direct', 'riesgo', 'risk_explanation', 'resumen_IA'
    ]
    try:
        worksheet = sheet_connection.worksheet(sheet_name); print(f"Reading master sheet: '{sheet_name}'")
        df = get_as_dataframe(worksheet, evaluate_formulas=True, header=0, na_filter=False, dtype=str) 
        
        # Ensure all expected columns exist, create if not
        for col_exp in all_expected_master_cols:
            if col_exp not in df.columns:
                print(f"Warning: Column '{col_exp}' not found in master. Creating it.")
                if col_exp in default_ai_cols: df[col_exp] = default_ai_cols[col_exp]
                elif col_exp == 'title_N': df[col_exp] = "0" # Default as string, will convert later
                else: df[col_exp] = "" # Default for other new/missing string cols
        
        # Specific handling for AI cols already present
        for col, default_val in default_ai_cols.items():
            if col in df.columns: df[col] = df[col].astype(str).replace('', default_val).fillna(default_val)
            # else: already created above with default_val
        
        if 'fecha_publicado' in df.columns: df['fecha_publicado'] = pd.to_datetime(df['fecha_publicado'], errors='coerce')
        else: df['fecha_publicado'] = pd.NaT # Should have been created by all_expected_master_cols
        if 'title_N' in df.columns: df['title_N'] = pd.to_numeric(df['title_N'], errors='coerce').fillna(0).astype(int)
        else: df['title_N'] = 0 # Should have been created

        # Ensure other string columns are indeed string and filled
        for col_str in ['enlace', 'pdf_link_direct', 'titulo', 'autores', 'resumen', 'topicos_str', 'tipos_str', 'publicador', 'formato', 'idioma', 'consecutivo', 'fecha_str', 'publicado_str', 'relaciones_str']:
            if col_str in df.columns: df[col_str] = df[col_str].astype(str).fillna("")
            else: df[col_str] = "" # Should have been created by all_expected_master_cols
            
        print(f"Read {len(df)} records from '{sheet_name}'"); return df, worksheet
    except gspread.exceptions.WorksheetNotFound:
        print(f"Master sheet '{sheet_name}' not found. Will define structure for new one.")
        empty_df_data = {}
        for col in all_expected_master_cols:
            if col == 'fecha_publicado': empty_df_data[col] = pd.Series(dtype='datetime64[ns]')
            elif col == 'title_N': empty_df_data[col] = pd.Series(dtype='int')
            elif col in default_ai_cols: empty_df_data[col] = default_ai_cols[col] # String "Not generated"
            else: empty_df_data[col] = pd.Series(dtype='str')
        return pd.DataFrame(empty_df_data), None 
    except Exception as e:
        print(f"Error reading master: {e}"); return pd.DataFrame(), None # Simpler fallback

# --- MAIN SCRIPT LOGIC ---
def main():
    print("Script started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    if not GOOGLE_API_SECRET or not GOOGLE_SHEET_ID: print("Secrets missing. Exiting."); return

    raw_dspace_df = get_dspace_data(URL)
    if raw_dspace_df.empty: print("No DSpace data. Exiting."); return
    base_df_from_dspace = process_dspace_records(raw_dspace_df) 
    if base_df_from_dspace.empty: print("No processable DSpace records. Exiting."); return

    augmented_dspace_df = base_df_from_dspace.copy()
    if 'pdf_link_direct' not in augmented_dspace_df.columns: augmented_dspace_df['pdf_link_direct'] = ""
    if 'riesgo' not in augmented_dspace_df.columns: augmented_dspace_df['riesgo'] = "Not generated"
    if 'risk_explanation' not in augmented_dspace_df.columns: augmented_dspace_df['risk_explanation'] = "Not generated"
    if 'resumen_IA' not in augmented_dspace_df.columns: augmented_dspace_df['resumen_IA'] = "Not generated"
    
    gs_connection = get_google_sheet_connection(GOOGLE_API_SECRET, GOOGLE_SHEET_ID)
    if not gs_connection: print("GSheet connection failed. Exiting."); return
    master_df, master_worksheet = get_master_sheet_data(gs_connection, MASTER_SHEET_NAME) 
    ALL_FINAL_COLUMNS = ['enlace', 'titulo', 'title_N', 'autores', 'fecha_publicado', 'resumen', 'topicos_str', 'tipos_str', 'publicador', 'formato', 'idioma', 'consecutivo', 'fecha_str', 'publicado_str', 'relaciones_str', 'pdf_link_direct', 'riesgo', 'risk_explanation', 'resumen_IA']

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

    if master_df.empty or not existing_enlaces_in_master: new_records_df_for_append = augmented_dspace_df.copy()
    else: new_records_df_for_append = augmented_dspace_df[~augmented_dspace_df['enlace'].astype(str).isin(existing_enlaces_in_master)].copy()
    print(f"Final count of new records for appending: {len(new_records_df_for_append)}")
    
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
            try: new_records_worksheet = gs_connection.worksheet(new_sheet_title); new_records_worksheet.clear(); print(f"Cleared dated sheet: '{new_sheet_title}'.")
            except gspread.exceptions.WorksheetNotFound: new_records_worksheet = gs_connection.add_worksheet(title=new_sheet_title, rows=max(1, len(final_df_for_dated_sheet) + 1), cols=len(cols_for_dated_sheet))
            for col_dt in final_df_for_dated_sheet.select_dtypes(include=['datetime64[ns]']).columns:
                final_df_for_dated_sheet[col_dt] = final_df_for_dated_sheet[col_dt].apply(lambda x: x.isoformat() if pd.notnull(x) and hasattr(x, 'isoformat') else "")
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
                if df_for_gs_append_final_vals: print(f"Appending {len(df_for_gs_append_final_vals)} rows to master."); master_worksheet.append_rows(df_for_gs_append_final_vals, value_input_option='USER_ENTERED'); print("Appended to master.")
                else: print("No data to append to master.")
                temp_master_df = master_df.copy(); temp_df_to_append = df_to_write_dated_sheet.copy()
                all_cols_concat = list(set(temp_master_df.columns) | set(temp_df_to_append.columns)); 
                if not all_cols_concat: all_cols_concat = ALL_FINAL_COLUMNS
                temp_master_df = temp_master_df.reindex(columns=all_cols_concat); temp_df_to_append = temp_df_to_append.reindex(columns=all_cols_concat)
                master_df = pd.concat([temp_master_df, temp_df_to_append], ignore_index=True)
                if 'fecha_publicado' in master_df.columns: master_df['fecha_publicado'] = pd.to_datetime(master_df['fecha_publicado'], errors='coerce')
                if 'title_N' in master_df.columns: master_df['title_N'] = pd.to_numeric(master_df['title_N'], errors='coerce').fillna(0).astype(int)
                for col_ai_c in ALL_FINAL_COLUMNS:
                    init_val_c = "Not generated" if col_ai_c in ['riesgo', 'risk_explanation', 'resumen_IA'] else (0 if col_ai_c == 'title_N' else (pd.NaT if col_ai_c == 'fecha_publicado' else ""))
                    if col_ai_c not in master_df.columns: master_df[col_ai_c] = init_val_c
                    else: 
                        if col_ai_c == 'fecha_publicado': master_df[col_ai_c] = pd.to_datetime(master_df[col_ai_c], errors='coerce').fillna(pd.NaT)
                        elif col_ai_c == 'title_N': master_df[col_ai_c] = pd.to_numeric(master_df[col_ai_c], errors='coerce').fillna(0).astype(int)
                        elif col_ai_c in ['riesgo', 'risk_explanation', 'resumen_IA']: master_df[col_ai_c] = master_df[col_ai_c].astype(str).fillna("Not generated")
                        else: master_df[col_ai_c] = master_df[col_ai_c].astype(str).fillna("")
                print(f"In-memory master_df updated. Length: {len(master_df)} (was {master_df_before_append_len})")
            except Exception as e_append: print(f"ERROR during append/master_df update: {e_append}")
        except Exception as e_dated: print(f"Error saving to dated sheet: {e_dated}")
    else: print("No new records to add.")

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
            
            document_text = None; print(f"  Attempting text extraction with PyMuPDF (fitz) for '{actual_pdf_to_process}'...")
            document_text = extract_text_from_pdf_fitz(actual_pdf_to_process)
            if not document_text:
                print(f"  PyMuPDF (fitz) failed. Attempting OCR fallback for '{actual_pdf_to_process}'...")
                try:
                    pytesseract.get_tesseract_version(); document_text = extract_text_from_pdf_ocr(actual_pdf_to_process, lang_code='spa')
                    if document_text: print("  OCR fallback extracted text.")
                    else: print("  OCR fallback also failed.")
                except Exception as e_tess: print(f"  OCR fallback skipped: Tesseract not working: {e_tess}")
            
            if document_text:
                print(f"  Extracted text (len {len(document_text)}). Gemini..."); riesgo_cat_val, riesgo_expl_val, resumen_ia_val = get_gemini_analysis(document_text)
                if str(master_df_updated.loc[index_loop, 'riesgo']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'riesgo'] = riesgo_cat_val
                if str(master_df_updated.loc[index_loop, 'risk_explanation']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'risk_explanation'] = riesgo_expl_val
                if str(master_df_updated.loc[index_loop, 'resumen_IA']).strip().lower() == "not generated": master_df_updated.loc[index_loop, 'resumen_IA'] = resumen_ia_val
                print(f"    R: {master_df_updated.loc[index_loop, 'riesgo']}, Expl: {master_df_updated.loc[index_loop, 'risk_explanation'][:50]}..., SumIA: {master_df_updated.loc[index_loop, 'resumen_IA'][:50]}...")
                processed_for_ai_count +=1
            else:
                print(f"  Failed text extraction from '{actual_pdf_to_process}'."); 
                for fld_ai_fail in ['riesgo', 'risk_explanation', 'resumen_IA']: 
                    if str(master_df_updated.loc[index_loop, fld_ai_fail]).strip().lower() == "not generated": master_df_updated.loc[index_loop, fld_ai_fail] = "PDF Text Extraction Failed (All Methods)"
            if processed_for_ai_count > 0 and processed_for_ai_count % 3 == 0 : print("  Pausing (3 docs)..."); time.sleep(5)
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

if __name__ == '__main__':
    main()