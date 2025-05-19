import requests
from bs4 import BeautifulSoup
import os
import time
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib3
import logging
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime
import threading
import unicodedata
from urllib.parse import urlparse

# Configure logging (minimal output)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class WebScraper:
    def __init__(self, debug=False):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.timeout = 20
        self.results = []
        self.error_links = []
        self.debug = debug
        self.processed_urls = set()  # Track processed URLs to avoid duplication
        # Configure session with retries
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.output_dir = f"output_{timestamp}"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            logging.info(f"Created output directory")

    def extract_links_from_file(self, file_path):
        """Read URLs from file"""
        logging.info(f"Reading links from {file_path}")
        links = []
        categories = {}
        current_category = "Uncategorized"
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                for i, line in enumerate(file, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    if line.startswith('# '):
                        current_category = line[2:].strip()
                        continue
                        
                    if line.startswith(('http://', 'https://')):
                        # Normalize URL to avoid duplicates
                        parsed = urlparse(line)
                        normalized_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
                        if not parsed.query:
                            normalized_url = normalized_url.rstrip('/')
                        links.append(normalized_url)
                        categories[normalized_url] = current_category
                    else:
                        logging.warning(f"Invalid URL at line {i}: {line}")
        
            logging.info(f"Extracted {len(links)} valid links")
            return links, categories
        except Exception as e:
            logging.error(f"Error reading {file_path}: {str(e)}")
            raise

    def sanitize_text(self, text):
        """Sanitize text while preserving Vietnamese characters and improving formatting"""
        if not text:
            return "No content"
        
        # Normalize Unicode to NFC
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1F\x7F]', ' ', text)
        
        # Improve paragraph formatting
        text = re.sub(r'\s*\n\s*', '\n\n', text)  # Normalize line breaks
        text = re.sub(r'([.!?])\s+', r'\1\n\n', text)  # Add breaks after sentences
        text = re.sub(r'(\S)\s+(\S)', r'\1 \2', text)  # Single space between words
        text = re.sub(r'\s+$', '', text)  # Remove trailing whitespace
        
        return text.strip()

    def get_page_content(self, url, category):
        """Extract text content from webpage with improved formatting"""
        # Check for duplicate URLs
        if url in self.processed_urls:
            logging.info(f"Skipping duplicate URL: {url}")
            return {
                'url': url,
                'status': 'skipped',
                'message': 'Duplicate URL'
            }
            
        self.processed_urls.add(url)
        logging.info(f"Scraping URL: {url}")
        timeout_event = threading.Event()

        def timeout_handler():
            timeout_event.set()

        timer = threading.Timer(30, timeout_handler)
        timer.start()

        try:
            response = self.session.get(url, headers=self.headers, timeout=self.timeout, verify=False)
            if timeout_event.is_set():
                raise TimeoutError("Processing timed out")
            
            if response.status_code == 200:
                response.encoding = response.apparent_encoding or 'utf-8'
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Check for JavaScript dependency
                if "javascript" in soup.get_text().lower() and not soup.find_all(['p', 'div', 'article']):
                    logging.warning(f"Page {url} may require JavaScript")

                # Extract meaningful content with better structure
                content_elements = soup.find_all(['p', 'div', 'article', 'section', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                text_parts = []
                current_section = ""
                
                for element in content_elements:
                    # Add section breaks for headings
                    if element.name.startswith('h'):
                        if current_section:
                            text_parts.append(current_section.strip())
                            current_section = ""
                        text_parts.append(f"\n\n=== {element.get_text(strip=True).upper()} ===\n\n")
                    else:
                        text = element.get_text(separator=" ", strip=True)
                        if text and len(text) > 20:
                            # Preserve natural paragraph breaks
                            current_section += text + "\n\n"
                
                if current_section:
                    text_parts.append(current_section.strip())
                
                full_text = "\n".join(text_parts)
                full_text = self.sanitize_text(full_text) if text_parts else "No meaningful content extracted"
                
                # Add category header
                if category and category != "Uncategorized":
                    full_text = f"=== CATEGORY: {category.upper()} ===\n\n{full_text}"
                
                result = {
                    'url': url,
                    'full_content': full_text,
                    'status': 'success',
                    'category': category
                }
                # Save immediately
                self.save_result_incrementally(result, len(self.results) + 1)
                return result
            else:
                error = {
                    'url': url,
                    'status': 'error',
                    'message': f"HTTP Error: {response.status_code}"
                }
                self.save_error_incrementally(error)
                return error
        except Exception as e:
            if timeout_event.is_set():
                error_message = "Processing timed out"
            else:
                error_message = str(e)
            error = {
                'url': url,
                'status': 'error',
                'message': error_message
            }
            self.save_error_incrementally(error)
            return error
        finally:
            timer.cancel()

    def save_result_incrementally(self, result, index):
        """Save only the text content incrementally with better organization"""
        try:
            # Create category subdirectory if it doesn't exist
            category_dir = os.path.join(self.output_dir, result.get('category', 'uncategorized'))
            os.makedirs(category_dir, exist_ok=True)
            
            # Generate filename from URL
            parsed_url = urlparse(result['url'])
            filename = f"{parsed_url.netloc}_{parsed_url.path.replace('/', '_')}".strip('_')
            filename = re.sub(r'[^\w\-_.]', '_', filename)[:150] + '.txt'
            
            content_file = os.path.join(category_dir, filename)
            
            with open(content_file, 'w', encoding='utf-8') as f:
                # Write metadata first
                f.write(f"URL: {result['url']}\n")
                f.write(f"CATEGORY: {result.get('category', 'uncategorized')}\n")
                f.write(f"SCRAPED AT: {datetime.now().isoformat()}\n")
                f.write("=" * 80 + "\n\n")
                
                # Write the formatted content
                f.write(result['full_content'])
            
            logging.info(f"Saved content for {result['url']} to {content_file}")
        except Exception as e:
            logging.error(f"Error saving content for {result['url']}: {str(e)}")

    def save_error_incrementally(self, error):
        """Save error incrementally"""
        try:
            errors_file = os.path.join(self.output_dir, "scraping_errors.txt")
            with open(errors_file, 'a', encoding='utf-8') as f:
                f.write(f"URL: {error['url']}\n")
                f.write(f"Error: {error.get('message', 'Unknown error')}\n")
                f.write("-" * 80 + "\n")
            
            logging.info(f"Saved error for {error['url']}")
        except Exception as e:
            logging.error(f"Error saving error for {error['url']}: {str(e)}")

    def scrape_links(self, links, categories, max_workers=5):
        """Scrape multiple links using multithreading with duplicate prevention"""
        # Remove duplicates while preserving order
        unique_links = []
        seen = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        total_links = len(unique_links)
        logging.info(f"Starting to scrape {total_links} unique links (after removing {len(links) - total_links} duplicates)")
        
        if self.debug:
            unique_links = unique_links[:3]
            max_workers = min(max_workers, 2)
            logging.info("Debug mode: Limited to 3 links")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(self.get_page_content, url, categories.get(url, "Uncategorized")): url 
                for url in unique_links
            }
            
            for i, future in enumerate(as_completed(future_to_url), 1):
                url = future_to_url[future]
                try:
                    data = future.result()
                    if data['status'] == 'success':
                        self.results.append(data)
                        logging.info(f"✓ Scraped ({i}/{total_links}): {url}")
                    elif data['status'] == 'skipped':
                        logging.info(f"↻ Skipped ({i}/{total_links}): {url} (duplicate)")
                    else:
                        self.error_links.append(data)
                        logging.warning(f"✗ Error ({i}/{total_links}): {url} - {data.get('message', 'Unknown error')}")
                    
                    progress = (i / total_links) * 100
                    print(f"Progress: {i}/{total_links} ({progress:.1f}%)")
                except Exception as e:
                    error = {
                        'url': url,
                        'status': 'error',
                        'message': str(e)
                    }
                    self.error_links.append(error)
                    self.save_error_incrementally(error)
                    logging.error(f"✗ Exception ({i}/{total_links}): {url} - {str(e)}")
        
        logging.info("Scraping completed")

    def generate_report(self):
        """Generate a detailed report"""
        logging.info(f"Generating report")
        try:
            report_file = os.path.join(self.output_dir, "report.md")
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write("# Web Scraping Report\n\n")
                f.write(f"## Summary\n")
                f.write(f"- **Total URLs processed**: {len(self.results) + len(self.error_links)}\n")
                f.write(f"- **Successful scrapes**: {len(self.results)}\n")
                f.write(f"- **Errors**: {len(self.error_links)}\n")
                f.write(f"- **Output directory**: `{self.output_dir}`\n\n")
                
                f.write("## Categories\n")
                categories = {}
                for result in self.results:
                    cat = result.get('category', 'Uncategorized')
                    categories[cat] = categories.get(cat, 0) + 1
                
                for cat, count in categories.items():
                    f.write(f"- {cat}: {count} pages\n")
                f.write("\n")
                
                f.write("## Error Details\n")
                for error in self.error_links:
                    f.write(f"### {error['url']}\n")
                    f.write(f"Error: {error.get('message', 'Unknown error')}\n\n")
            
            logging.info(f"Generated comprehensive report")
            print(f"\nReport saved to {report_file}")
        except Exception as e:
            logging.error(f"Error generating report: {str(e)}")
            print(f"Error generating report: {str(e)}")

def main():
    try:
        links_file = "links.txt"
        if not os.path.exists(links_file):
            logging.error(f"Links file not found: {links_file}")
            print(f"Error: Links file not found: {links_file}")
            return

        scraper = WebScraper(debug=False)
        links, categories = scraper.extract_links_from_file(links_file)
        if not links:
            logging.warning("No valid links to scrape")
            print("Error: No valid links to scrape")
            return

        scraper.scrape_links(links, categories, max_workers=5)
        scraper.generate_report()
        
        print(f"\nResults saved in {scraper.output_dir}")
        print(f"Total unique URLs: {len(scraper.results) + len(scraper.error_links)}")
        print(f"Successful: {len(scraper.results)}")
        print(f"Errors: {len(scraper.error_links)}")
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        print(f"Error: {str(e)}")
    finally:
        logging.info("Script completed")

if __name__ == "__main__":
    main()