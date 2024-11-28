import requests
from bs4 import BeautifulSoup
import os
import io
import pdfplumber
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
import queue
import random
import time
from urllib.parse import urljoin, urlparse
from threading import Thread
from datetime import datetime
import hashlib

# Initialize logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Cache for saved content
saved_content_hashes = set()

# Clean text
def clean_text(text):
    text = ' '.join(text.split())
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# Extract and clean
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def extract_content(url):
    try:
        response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        paragraphs = soup.find_all('p')
        content = '\n\n'.join([p.get_text() for p in paragraphs])

        return clean_text(content)
    except (requests.RequestException, requests.ConnectionError, requests.Timeout) as e:
        logging.error(f"Failed to retrieve {url}: {e}")
        return None

# Extract PDFs
def extract_pdf_content(url):
    try:
        response = requests.get(url, timeout=10)  # Add timeout to prevent hanging
        response.raise_for_status()
        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            content = ''
            for page in pdf.pages:
                content += page.extract_text() + '\n'

        return clean_text(content)
    except (requests.RequestException, requests.ConnectionError, requests.Timeout) as e:
        logging.error(f"Failed to retrieve PDF {url}: {e}")
        return None

# Save to markdown
def save_to_markdown(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.info(f"Saved {filename}")

# Check if content is already saved
def is_content_saved(content):
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    if content_hash in saved_content_hashes:
        return True
    saved_content_hashes.add(content_hash)
    return False

# Scrape and clean
def scrape_and_clean(start_url, max_depth, max_files, chunk_size, rate_limit_min=1, rate_limit_max=6):
    visited_urls = set()
    to_visit = queue.Queue()
    to_visit.put((start_url, 0))
    base_url = "{0.scheme}://{0.netloc}".format(urlparse(start_url))
    file_counter = 1

    while not to_visit.empty() and file_counter <= max_files:
        url, depth = to_visit.get()
        if url in visited_urls or depth > max_depth:
            continue

        logging.info(f"Visiting: {url} (depth {depth})")
        visited_urls.add(url)

        content = extract_content(url) if not url.endswith('.pdf') else extract_pdf_content(url)

        if content and not is_content_saved(content):
            # Save chunks as markdown
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                filename = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_chunk_{file_counter}.md"
                save_to_markdown(filename, chunk)
                file_counter += 1

        # Extract and queue Links
        try:
            soup = BeautifulSoup(requests.get(url).text, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    # Link validation
                    if full_url not in visited_urls and full_url.startswith(base_url):
                        to_visit.put((full_url, depth + 1))
                        logging.info(f"Queued: {full_url}")
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")

        # Rate limit
        time.sleep(random.uniform(rate_limit_min, rate_limit_max))

    logging.info("Scraping finished.")

# Threaded scraping
def threaded_scrape_and_clean(start_url, max_depth, max_files, chunk_size, num_threads=5):
    threads = []
    for _ in range(num_threads):
        thread = Thread(target=scrape_and_clean, args=(start_url, max_depth, max_files, chunk_size))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    logging.info("All threads have completed.")

if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Reinforcement_learning"
    max_depth = 2
    max_files = 5000
    chunk_size = 5000
    num_threads = 5

    logging.info("Starting scraping process...")
    threaded_scrape_and_clean(start_url, max_depth, max_files, chunk_size, num_threads)
