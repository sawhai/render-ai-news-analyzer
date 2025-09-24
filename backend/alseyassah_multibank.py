#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Al-Seyassah Multi-Bank Analysis Script with Vision API Integration
Adopts Al-Rai's Vision API methodology while keeping Al-Seyassah's download process intact
"""

import os
import sys
import asyncio
import nest_asyncio
from dotenv import load_dotenv
from datetime import datetime, timedelta
from playwright.async_api import async_playwright
import requests
import fitz  # PyMuPDF
import re
import json
import base64
from openai import OpenAI
from pathlib import Path

# Import the generated bank patterns
try:
    from bank_patterns import BANK_CONFIGS, get_bank_info
except ImportError:
    print("Warning: bank_patterns.py not found. Please run the configuration generator first.")
    
    # Fallback bank configuration
    BANK_CONFIGS = {
        'gulf_bank': {'english_name': 'Gulf Bank', 'arabic_name': 'بنك الخليج'},
        'nbk': {'english_name': 'National Bank of Kuwait', 'arabic_name': 'البنك الوطني الكويتي'},
        'kfh': {'english_name': 'Kuwait Finance House', 'arabic_name': 'بيت التمويل الكويتي'},
        'boubyan_bank': {'english_name': 'Boubyan Bank', 'arabic_name': 'بنك بوبيان'},
        'cbk': {'english_name': 'Commercial Bank of Kuwait', 'arabic_name': 'البنك التجاري الكويتي'},
        'burgan_bank': {'english_name': 'Burgan Bank', 'arabic_name': 'بنك برقان'},
        'kib': {'english_name': 'Kuwait International Bank', 'arabic_name': 'بنك الكويت الدولي'},
        'abk': {'english_name': 'Al Ahli Bank of Kuwait', 'arabic_name': 'البنك الأهلي الكويتي'},
        'warba_bank': {'english_name': 'Warba Bank', 'arabic_name': 'بنك وربة'}
    }
    
    def get_bank_info(bank_code):
        return BANK_CONFIGS.get(bank_code, {'english_name': bank_code, 'arabic_name': bank_code})

# Setup
nest_asyncio.apply()
load_dotenv()

# Validate required API keys
if not os.getenv('OPENAI_API_KEY'):
    raise ValueError('OPENAI_API_KEY is not set - required for Vision API')

# Keep original Al-Seyassah download directory structure
BASE_DATA_DIR = os.getenv("DATA_DIR", "/tmp/bank_news_data")
download_dir = Path(BASE_DATA_DIR) / "AlSeyassah"
os.makedirs(download_dir, exist_ok=True)

# Bank configurations for analysis
ACTIVE_BANKS = ['gulf_bank', 'nbk', 'kfh', 'cbk', 'burgan_bank', 'kib', 'abk', 'warba_bank', 'boubyan_bank']

# Vision API Analyzer Class - ADOPTED FROM AL-RAI
class VisionBankAnalyzer:
    def __init__(self, openai_api_key=None, model="gpt-4o"):
        self.api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        self.known_banks = {
            'gulf_bank': {'ar': 'بنك الخليج', 'en': 'Gulf Bank'},
            'nbk': {'ar': 'البنك الوطني الكويتي', 'en': 'National Bank of Kuwait'},
            'kfh': {'ar': 'بيت التمويل الكويتي', 'en': 'Kuwait Finance House'},
            'boubyan_bank': {'ar': 'بنك بوبيان', 'en': 'Boubyan Bank'},
            'cbk': {'ar': 'البنك التجاري الكويتي', 'en': 'Commercial Bank of Kuwait'},
            'burgan_bank': {'ar': 'بنك برقان', 'en': 'Burgan Bank'},
            'kib': {'ar': 'بنك الكويت الدولي', 'en': 'Kuwait International Bank'},
            'abk': {'ar': 'البنك الأهلي الكويتي', 'en': 'Al Ahli Bank of Kuwait'},
            'warba_bank': {'ar': 'بنك وربة', 'en': 'Warba Bank'}
        }

    def extract_page_as_image(self, pdf_path, page_num, dpi=200):
        """Extract a single page as PNG image and raw text"""
        doc = fitz.open(pdf_path)
        if page_num >= len(doc):
            raise ValueError(f"Page {page_num} does not exist")
        
        page = doc.load_page(page_num)
        raw_text = page.get_text("text")
        pix = page.get_pixmap(dpi=dpi)
        png_bytes = pix.tobytes("png")
        doc.close()
        return png_bytes, raw_text
    
    def create_analysis_prompt(self):
        """Enhanced prompt with stricter detection criteria"""
        return """
You are analyzing this newspaper page for PRIMARY banking news stories about Kuwaiti banks.

STRICT DETECTION RULES:
- ONLY detect content where a Kuwaiti bank is the MAIN SUBJECT of a news story
- IGNORE casual mentions, facilitation roles, or statistical inclusions
- IGNORE general market news that just mentions banks
- IGNORE legal notices unless they're major bank announcements

REQUIRED FOR DETECTION:
1. Dedicated headline or article section about the specific bank
2. Substantive information about bank's own activities/results/announcements
3. Bank as active participant, not passive mention

Extract ONLY genuine bank news and return ONLY valid JSON:

{
  "page_analysis": {
    "page_has_banking_content": true/false,
    "analysis_confidence": 0.0-1.0,
    "content_quality": "primary_subject/secondary_mention/casual_reference"
  },
  "banks_found": [
    {
      "bank_code": "detected_identifier", 
      "bank_name_ar": "Arabic name found",
      "bank_name_en": "English equivalent",
      "content_type": "news_article/announcement/financial_results",
      "headline": "main headline mentioning bank",
      "summary": "what the bank specifically did/announced",
      "key_details": ["specific bank actions", "measurable outcomes", "dates"],
      "confidence": 0.0-1.0,
      "is_primary_subject": true/false,
      "evidence_strength": "strong/medium/weak"
    }
  ]
}

IMPORTANT INSTRUCTIONS:
- Translate ALL text content to English
- Include headlines in English 
- Write summaries in English explaining what the BANK specifically did
- Include specific details like amounts, dates, events related to BANK actions
- Look for ANY mention of these Kuwaiti banks: Gulf Bank, National Bank of Kuwait, Kuwait Finance House, Boubyan Bank, Commercial Bank of Kuwait, Burgan Bank, Kuwait International Bank, Al Ahli Bank of Kuwait, Warba Bank
- ONLY include banks that are PRIMARY SUBJECTS of dedicated news content
- Set is_primary_subject=true ONLY if bank is main focus of the story
- Return ONLY valid JSON with English content
"""


    async def analyze_page_with_vision(self, png_bytes, raw_text):
        """Send page image and text to Vision API for analysis"""
        try:
            png_b64 = base64.b64encode(png_bytes).decode("utf-8")
            data_url = f"data:image/png;base64,{png_b64}"
            
            instruction = self.create_analysis_prompt()
            
            user_content = [
                {"type": "text", "text": instruction},
                {"type": "text", "text": f"Context: {raw_text[:5000]}"},
                {"type": "image_url", "image_url": {"url": data_url}}
            ]
            
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.model,
                messages=[{"role": "user", "content": user_content}],
                temperature=0.1,
                max_tokens=2000
            )
            
            result_text = response.choices[0].message.content.strip()
            
            try:
                result_data = json.loads(result_text)
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    raise ValueError("Could not parse JSON")
            
            return result_data
            
        except Exception as e:
            print(f"Vision API error: {e}")
            return {
                "page_analysis": {"page_has_banking_content": False, "analysis_confidence": 0.0},
                "banks_found": [],
                "error": str(e)
            }

    def map_to_bank_code(self, detected_name):
        """Map detected bank name to internal bank code"""
        detected_lower = detected_name.lower()
        
        for bank_code, info in self.known_banks.items():
            if info['ar'] in detected_name or info['en'].lower() in detected_lower:
                return bank_code
        return None

    def save_debug_text(self, raw_text, page_idx, pdf_path, today, debug_dir):
        """Save extracted text to debug file for inspection"""
        debug_filename = f"page_{page_idx + 1}_extracted_text_{today.strftime('%Y%m%d_%H%M%S')}.txt"
        debug_filepath = os.path.join(debug_dir, debug_filename)
        
        try:
            with open(debug_filepath, 'w', encoding='utf-8') as debug_file:
                debug_file.write(f"=== PAGE {page_idx + 1} TEXT EXTRACTION DEBUG ===\n")
                debug_file.write(f"Extraction Date: {today.strftime('%Y-%m-%d %H:%M:%S')}\n")
                debug_file.write(f"PDF Source: {pdf_path}\n")
                debug_file.write(f"Page Number: {page_idx + 1} (0-indexed: {page_idx})\n")
                debug_file.write(f"Text Length: {len(raw_text)} characters\n")
                debug_file.write(f"Text Lines: {len(raw_text.splitlines())} lines\n")
                debug_file.write("=" * 80 + "\n\n")
                debug_file.write("FULL EXTRACTED TEXT:\n")
                debug_file.write("-" * 40 + "\n")
                debug_file.write(raw_text)
                debug_file.write("\n" + "-" * 40 + "\n")
                
                # Add text analysis section
                debug_file.write("\nTEXT ANALYSIS:\n")
                has_arabic = any(ord(char) > 1536 and ord(char) < 1791 for char in raw_text)
                has_english = any(char.isascii() and char.isalpha() for char in raw_text)
                debug_file.write(f"Contains Arabic text: {'Yes' if has_arabic else 'No'}\n")
                debug_file.write(f"Contains English text: {'Yes' if has_english else 'No'}\n")
                debug_file.write(f"Word count (approximate): {len(raw_text.split())}\n")
                debug_file.write(f"Line count: {len(raw_text.splitlines())}\n")
                
                # Check for potential bank mentions
                bank_keywords = ['بنك', 'Bank', 'NBK', 'KFH', 'Gulf', 'الخليج', 'الوطني', 'التمويل', 'بوبيان', 'KIB', 'CBK', 'ABK', 'Warba', 'وربة', 'برقان', 'Burgan']
                found_keywords = [keyword for keyword in bank_keywords if keyword in raw_text]
                debug_file.write(f"Bank keywords found: {found_keywords if found_keywords else 'None'}\n")
                
                # Add character encoding info
                debug_file.write(f"Text encoding: UTF-8\n")
                debug_file.write(f"Non-ASCII characters: {sum(1 for char in raw_text if not char.isascii())}\n")
            
            print(f"      DEBUG: Text saved to {debug_filename}")
            return debug_filepath
            
        except Exception as debug_error:
            print(f"      WARNING: Could not save debug text file: {debug_error}")
            return None

    async def analyze_pdf_pages(self, pdf_path, start_page=4, end_page=9):
        """Main analysis function - ANALYZE PAGES 5-10 (indices 4-9) like Al-Seyassah"""
        print("VISION API ANALYSIS STARTED")
        print(f"   PDF: {pdf_path}")
        print(f"   Pages: {start_page + 1} to {end_page + 1}")
        print(f"   API Key: {self.api_key[:10]}...{self.api_key[-4:]}")
        
        # Create debug directory for text extraction
        debug_dir = os.path.join(os.path.dirname(pdf_path), "debug_text_extraction")
        os.makedirs(debug_dir, exist_ok=True)
        print(f"   Debug text files will be saved to: {debug_dir}")
        
        all_bank_findings = {}
        today = datetime.now()
        
        try:
            for page_idx in range(start_page, end_page + 1):
                print(f"\n   ANALYZING PAGE {page_idx + 1}")
                
                try:
                    # Extract page content
                    png_bytes, raw_text = self.extract_page_as_image(pdf_path, page_idx)
                    text_preview = raw_text[:200].replace('\n', ' ')
                    print(f"      Text preview: {text_preview}...")
                    print(f"      Image size: {len(png_bytes)} bytes")
                    print(f"      Extracted text length: {len(raw_text)} characters")
                    
                    # Save debug text file
                    debug_filepath = self.save_debug_text(raw_text, page_idx, pdf_path, today, debug_dir)
                    
                    # Save screenshot
                    screenshot_path = f"temp_page_{page_idx + 1}_{today.strftime('%H%M%S')}.png"
                    with open(screenshot_path, 'wb') as f:
                        f.write(png_bytes)
                    
                    # Call Vision API
                    print(f"      Calling Vision API...")
                    page_result = await self.analyze_page_with_vision(png_bytes, raw_text)
                    
                    # Log raw Vision API response
                    print(f"      RAW VISION API RESPONSE:")
                    print(f"      {json.dumps(page_result, indent=2, ensure_ascii=False)}")
                    
                    # Log Vision API response summary
                    banks_found = len(page_result.get('banks_found', []))
                    has_content = page_result.get('page_analysis', {}).get('page_has_banking_content', False)
                    confidence = page_result.get('page_analysis', {}).get('analysis_confidence', 0.0)
                    
                    print(f"      VISION API RESPONSE SUMMARY:")
                    print(f"         Banking content: {has_content}")
                    print(f"         Banks found: {banks_found}")
                    print(f"         Confidence: {confidence:.2f}")
                    
                    if page_result.get('error'):
                        print(f"      API Error: {page_result['error']}")
                    
                    # Log detailed findings with mapping attempts
                    for i, bank_finding in enumerate(page_result.get('banks_found', []), 1):
                        bank_name = bank_finding.get('bank_name_ar') or bank_finding.get('bank_name_en', 'Unknown')
                        print(f"         {i}. {bank_name}")
                        print(f"            Headline: {bank_finding.get('headline', 'N/A')}")
                        print(f"            Content Type: {bank_finding.get('content_type', 'N/A')}")
                        print(f"            Confidence: {bank_finding.get('confidence', 0.0):.2f}")
                    
                    # Process findings into bank results
                    for bank_finding in page_result.get('banks_found', []):
                        detected_name_ar = bank_finding.get('bank_name_ar', '')
                        detected_name_en = bank_finding.get('bank_name_en', '')
                        
                        # Log bank mapping attempts
                        print(f"         BANK MAPPING ATTEMPT:")
                        print(f"            Arabic name detected: '{detected_name_ar}'")
                        print(f"            English name detected: '{detected_name_en}'")
                        
                        # Try mapping with Arabic name first
                        bank_code = None
                        if detected_name_ar:
                            bank_code = self.map_to_bank_code(detected_name_ar)
                            print(f"            Mapping Arabic name '{detected_name_ar}' -> {bank_code}")
                        
                        # If no match, try English name
                        if not bank_code and detected_name_en:
                            bank_code = self.map_to_bank_code(detected_name_en)
                            print(f"            Mapping English name '{detected_name_en}' -> {bank_code}")
                        
                        # Final result
                        if bank_code:
                            print(f"            ✅ SUCCESSFULLY MAPPED TO: {bank_code}")
                        else:
                            print(f"            ❌ FAILED TO MAP - No matching bank code found")
                            print(f"            Available bank codes: {list(self.known_banks.keys())}")
                            print(f"            Known Arabic names: {[info['ar'] for info in self.known_banks.values()]}")
                            print(f"            Known English names: {[info['en'] for info in self.known_banks.values()]}")
                        
                        if bank_code:
                            # Initialize bank result if not exists
                            if bank_code not in all_bank_findings:
                                bank_info = get_bank_info(bank_code)
                                all_bank_findings[bank_code] = {
                                    'newspaper_name': 'Al-Seyassah',
                                    'bank_name': bank_info['english_name'],
                                    'bank_code': bank_code,
                                    'bank_found': True,
                                    'relevant_pages': [],
                                    'analysis_text': '',
                                    'page_screenshots': {},
                                    'date': today,
                                    'summary': '',
                                    'highlights': []
                                }
                            
                            # Add page data
                            if page_idx not in all_bank_findings[bank_code]['relevant_pages']:
                                all_bank_findings[bank_code]['relevant_pages'].append(page_idx)
                                all_bank_findings[bank_code]['page_screenshots'][page_idx] = screenshot_path
                            
                            # Add headline as highlight
                            if bank_finding.get('headline'):
                                all_bank_findings[bank_code]['highlights'].append(bank_finding['headline'])
                            
                            # Append to analysis text
                            all_bank_findings[bank_code]['analysis_text'] += f"\n### Page {page_idx + 1}\n{bank_finding.get('summary', '')}\n"
                            
                            # Set summary if empty
                            if not all_bank_findings[bank_code]['summary']:
                                all_bank_findings[bank_code]['summary'] = bank_finding.get('summary', '')
                    
                except Exception as e:
                    print(f"      ERROR analyzing page {page_idx + 1}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\nVISION API ANALYSIS SUMMARY:")
            print(f"   Banks with content: {len(all_bank_findings)}")
            print(f"   Debug text files saved to: {debug_dir}")
            
            # Print summary of findings
            for bank_code, result in all_bank_findings.items():
                pages = [p+1 for p in result['relevant_pages']]
                highlights_count = len(result['highlights'])
                print(f"   {result['bank_name']}: {len(result['relevant_pages'])} pages, {highlights_count} highlights (Pages: {pages})")
            
            return all_bank_findings
            
        except Exception as e:
            print(f"FATAL ERROR in Vision API: {e}")
            import traceback
            traceback.print_exc()
            return {}

# Helper functions from Al-Rai
def extract_summary_from_analysis(analysis_text):
    """Extract summary from AI analysis text"""
    summary_match = re.search(r'## OVERALL SUMMARY\s*\n(.*?)(?=\n##|\Z)', analysis_text, re.DOTALL)
    if summary_match:
        return summary_match.group(1).strip()
    
    # Fallback to regular SUMMARY section
    summary_match = re.search(r'## SUMMARY\s*\n(.*?)(?=\n##|\Z)', analysis_text, re.DOTALL)
    if summary_match:
        return summary_match.group(1).strip()
    
    if not analysis_text or len(analysis_text) < 20:
        return "No summary available"
    return analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text

def extract_highlights_from_analysis(analysis_text):
    """Extract highlights from AI analysis text"""
    if not analysis_text:
        return []
    
    highlights = []
    
    # Look for KEY HIGHLIGHTS section
    highlights_match = re.search(r'## KEY HIGHLIGHTS\s*\n(.*?)(?=\n##|\Z)', analysis_text, re.DOTALL)
    if highlights_match:
        highlights_text = highlights_match.group(1).strip()
        
        for line in highlights_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                headline = line.strip('- •').strip()
                if len(headline) > 20:
                    highlights.append(headline)
    
    # Fallback: Look for KEY POINTS section or simple bullet parsing
    if not highlights:
        for line in analysis_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                headline = line.strip('- •').strip()
                if len(headline) > 20:
                    highlights.append(headline)
    
    return highlights[:5]  # Max 5 highlights

# KEEP ORIGINAL AL-SEYASSAH DOWNLOAD METHOD UNCHANGED
async def download_alseyassah_pdf():
    """Download Al-Seyassah PDF using the original proven method"""
    today = datetime.now()
    pdf_path = os.path.join(download_dir, f"AlSeyassah_Media_{today.strftime('%Y-%m-%d')}.pdf")
    
    print("📱 Downloading Al-Seyassah newspaper...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await context.new_page()
        
        try:
            archive_url = "https://alseyassah.com/%d8%b9%d8%af%d8%af-%d8%a7%d9%84%d9%8a%d9%88%d9%85/"
            await page.goto(archive_url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(2)
            
            today_formatted = today.strftime('%Y-%m-%d')
            
            recent_edition = await page.evaluate(f"""() => {{
                const todayDate = '{today_formatted}';
                const candidates = [];
                
                const allElements = Array.from(document.querySelectorAll('*'));
                
                for (const el of allElements) {{
                    const text = el.textContent || el.alt || el.title || '';
                    const href = el.href || el.getAttribute('onclick') || '';
                    
                    if (text.includes(todayDate) || href.includes(todayDate)) {{
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 50 && rect.height > 50) {{
                            candidates.push({{
                                x: rect.x + rect.width/2,
                                y: rect.y + rect.height/2,
                                href: href,
                                text: text.slice(0, 100),
                                priority: 0
                            }});
                        }}
                    }}
                    else if ((el.tagName === 'IMG' || el.tagName === 'A') && 
                             (href.includes('article') || href.includes('2025') || text.includes('2025'))) {{
                        const rect = el.getBoundingClientRect();
                        if (rect.width > 100 && rect.height > 100) {{
                            candidates.push({{
                                x: rect.x + rect.width/2,
                                y: rect.y + rect.height/2,
                                href: href,
                                text: text.slice(0, 100),
                                priority: 1
                            }});
                        }}
                    }}
                }}
                
                candidates.sort((a, b) => {{
                    if (a.priority !== b.priority) {{
                        return a.priority - b.priority;
                    }}
                    return b.x - a.x;
                }});
                
                return candidates.length > 0 ? candidates[0] : null;
            }}""")
            
            if recent_edition:
                print(f"✅ Found recent edition: {recent_edition['href']}")
                await page.mouse.click(recent_edition['x'], recent_edition['y'])
                await page.wait_for_load_state("networkidle", timeout=15000)
                await asyncio.sleep(2)
                
                page_content = await page.content()
                
                pdf_patterns = [
                    r'https?://[^"\s]+\.pdf[^"\s]*',
                    r'/[^"\s]*\.pdf[^"\s]*',
                ]
                
                found_pdf_urls = []
                for pattern in pdf_patterns:
                    matches = re.findall(pattern, page_content, re.IGNORECASE)
                    found_pdf_urls.extend(matches)
                
                unique_urls = list(set(found_pdf_urls))
                
                for url in unique_urls:
                    if url.startswith('/'):
                        url = 'https://alseyassah.com' + url
                    
                    if 'alseyassah.com' in url:
                        print(f"🔗 Trying PDF URL: {url}")
                        
                        try:
                            response = requests.get(url, stream=True, timeout=30)
                            if response.status_code == 200:
                                content_type = response.headers.get('content-type', '').lower()
                                if 'pdf' in content_type:
                                    with open(pdf_path, 'wb') as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                    print(f"✅ PDF downloaded: {os.path.getsize(pdf_path)} bytes")
                                    
                                    # Verify PDF size
                                    file_size = os.path.getsize(pdf_path)
                                    if file_size > 1000000:  # At least 1MB
                                        print(f"✅ Downloaded substantial PDF ({file_size} bytes)")
                                        break
                                    else:
                                        print(f"⚠️ Small PDF ({file_size} bytes) - continuing search")
                        except Exception as e:
                            print(f"⚠️ Error downloading: {e}")
                            continue
            
        except Exception as e:
            print(f"⚠️ Error during download: {e}")
        
        finally:
            await browser.close()
    
    if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 10000:
        try:
            doc = fitz.open(pdf_path)
            page_count = len(doc)
            doc.close()
            print(f"✅ PDF verified: {page_count} pages")
            return pdf_path
        except:
            pass
    
    return None

def create_empty_result(bank_code, today, pdf_path=None, error_msg="No content found"):
    """Create an empty result structure for a single bank"""
    bank_info = get_bank_info(bank_code)
    return {
        'newspaper_name': 'Al-Seyassah',
        'bank_name': bank_info['english_name'],
        'bank_code': bank_code,
        'bank_found': False,
        'relevant_pages': [],
        'analysis_text': f"No mentions of {bank_info['english_name']} found in Al-Seyassah. {error_msg}",
        'page_screenshots': {},
        'date': today,
        'summary': "No content found",
        'highlights': [],
        'pdf_path': pdf_path
    }

async def analyze_all_banks(selected_model="gpt-4o"):
    """
    Main multi-bank analysis function with Vision API - ADAPTED FROM AL-RAI
    """
    today = datetime.now() 
    print(f"Starting Al-Seyassah Vision API multi-bank analysis...")
    print(f"Date: {today.strftime('%Y-%m-%d')}")
    print(f"Banks to analyze: {', '.join([get_bank_info(bank)['english_name'] for bank in ACTIVE_BANKS])}")
    print("=" * 60)
    
    # Step 1: Download PDF using ORIGINAL AL-SEYASSAH method
    pdf_path = await download_alseyassah_pdf()
    
    if not pdf_path:
        print("Failed to obtain PDF. Aborting multi-bank analysis.")
        return {bank_code: create_empty_result(bank_code, today, None, "PDF download failed") for bank_code in ACTIVE_BANKS}
    
    # Step 2: Check if OpenAI API key is available
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        print("WARNING: OPENAI_API_KEY not found. Cannot use Vision API.")
        return {bank_code: create_empty_result(bank_code, today, pdf_path, "OpenAI API key not configured") for bank_code in ACTIVE_BANKS}
    
    print(f"OpenAI API key found: {openai_api_key[:10]}...")
    
    # Step 3: Validate PDF structure
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        print(f"PDF validation: {total_pages} pages found")
        
        if total_pages < 10:
            print(f"WARNING: PDF only has {total_pages} pages, but trying to analyze pages 5-10")
        
        # Check if pages 5-10 exist and have content
        for page_idx in range(4, min(10, total_pages)):
            page = doc.load_page(page_idx)
            text_length = len(page.get_text("text"))
            print(f"   Page {page_idx + 1}: {text_length} characters of text")
            
        doc.close()
    except Exception as e:
        print(f"PDF validation error: {e}")
        return {bank_code: create_empty_result(bank_code, today, pdf_path, f"PDF validation failed: {str(e)}") for bank_code in ACTIVE_BANKS}
    
    # Step 4: Use Vision API to analyze pages 5-10 (indices 4-9)
    print(f"Starting Vision API analysis of pages 5-10 from {pdf_path}")
    
    vision_results = {}
    try:
        analyzer = VisionBankAnalyzer(openai_api_key,selected_model)
        vision_results = await analyzer.analyze_pdf_pages(pdf_path, 4, min(9, total_pages-1))
        
        print(f"Vision API analysis completed. Found content for {len(vision_results)} banks.")
        
    except Exception as e:
        print(f"Error in Vision API analysis: {str(e)}")
        print("Falling back to empty results...")
        vision_results = {}
    
    # Step 5: Process results for each bank
    bank_results = {}
    
    for bank_code in ACTIVE_BANKS:
        try:
            if bank_code in vision_results:
                result = vision_results[bank_code]
                result['pdf_path'] = pdf_path
                
                # Ensure highlights are properly formatted
                if not result.get('highlights'):
                    result['highlights'] = extract_highlights_from_analysis(result.get('analysis_text', ''))
                
                # Ensure summary is present
                if not result.get('summary') or result['summary'] == "No summary available":
                    result['summary'] = extract_summary_from_analysis(result.get('analysis_text', ''))
                
                bank_results[bank_code] = result
                
                bank_found = result.get('bank_found', False)
                relevant_pages = result.get('relevant_pages', [])
                bank_name = get_bank_info(bank_code)['english_name']
                
                print(f"Processing Vision API results for {bank_name}...")
                print(f"   {bank_name}: {'Content found' if bank_found else 'No content'}")
                if bank_found:
                    print(f"   Pages: {[p+1 for p in relevant_pages]}")
                    print(f"   Highlights: {len(result.get('highlights', []))}")
            else:
                bank_results[bank_code] = create_empty_result(bank_code, today, pdf_path)
                print(f"   {get_bank_info(bank_code)['english_name']}: No content found")
            
        except Exception as e:
            print(f"Error analyzing {get_bank_info(bank_code)['english_name']}: {e}")
            bank_results[bank_code] = create_empty_result(bank_code, today, pdf_path, f"Analysis error: {str(e)}")
    
    # Step 6: Clean up PDF (screenshots are handled by Vision API)
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
            print("Removed temporary PDF")
    except Exception as e:
        print(f"Warning during PDF cleanup: {e}")
    
    # Step 7: Print summary
    print("\n" + "=" * 60)
    print("AL-SEYASSAH VISION API MULTI-BANK ANALYSIS SUMMARY")
    print("=" * 60)
    
    for bank_code, result in bank_results.items():
        status = "CONTENT FOUND" if result['bank_found'] else "NO CONTENT"
        pages_info = f" (Pages: {[p+1 for p in result['relevant_pages']]})" if result['relevant_pages'] else ""
        print(f"{status}: {result['bank_name']}{pages_info}")
    
    return bank_results

async def main():
    """Main entry point for testing"""
    print("Al-Seyassah Vision API Multi-Bank Analysis Started...")
    
    # Test mode
    if len(sys.argv) > 1 and sys.argv[1] == "--test-vision":
        print("\n=== VISION API TEST MODE ===")
        try:
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                print("ERROR: OPENAI_API_KEY not found")
                return
            
            print(f"API Key found: {api_key[:10]}...{api_key[-4:]}")
            
            client = OpenAI(api_key=api_key)
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": "Say 'Vision API test successful'"}],
                max_tokens=10
            )
            
            test_response = response.choices[0].message.content
            print(f"Test response: {test_response}")
            print("Vision API setup working!")
            
        except Exception as e:
            print(f"Vision API test failed: {e}")
        return
    
    # Full analysis
    try:
        results = await analyze_all_banks()
        
        print("\nVision API multi-bank analysis completed!")
        print(f"Results summary:")
        
        for bank_code, result in results.items():
            print(f"\n{result['bank_name']}:")
            print(f"   Date: {result['date'].strftime('%Y-%m-%d')}")
            print(f"   Content Found: {result['bank_found']}")
            if result['relevant_pages']:
                print(f"   Relevant Pages: {[p+1 for p in result['relevant_pages']]}")
                print(f"   Highlights: {len(result['highlights'])} items")
            print(f"   Summary Preview: {result['summary'][:100]}...")
            
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()

# Compatibility functions for master orchestrator integration
async def analyze_for_gulf_bank():
    """Compatibility function for Gulf Bank analysis"""
    results = await analyze_all_banks()
    return results.get('gulf_bank', {})

async def analyze_for_nbk():
    """NBK analysis function"""
    results = await analyze_all_banks()
    return results.get('nbk', {})

async def analyze_for_kfh():
    """KFH analysis function"""
    results = await analyze_all_banks()
    return results.get('kfh', {})

# Run the script (for testing only)
if __name__ == "__main__":
    print("Vision API Script execution started...")
    asyncio.run(main())
    print("Vision API Script execution completed.")