import aiohttp
import asyncio
from aiohttp import ClientSession
from bs4 import BeautifulSoup
import os
import io
import pdfplumber
import logging
import hashlib
import random
import time
from urllib.parse import urljoin, urldefrag, urlparse
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential
import threading
import queue

# Initialize logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread-safe data structures
visited_urls = set()
visited_urls_lock = threading.Lock()

content_hashes = set()
content_hashes_lock = threading.Lock()

# Async semaphore for rate limiting
semaphore = asyncio.Semaphore(5)  # Limit concurrent requests

# Normalize URLs to avoid duplicates
def normalize_url(url):
    url, _ = urldefrag(url)  # Remove fragment
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"

# Clean extracted text
def clean_text(text):
    text = ' '.join(text.split())
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# Save content to a Markdown file
def save_to_markdown(filename, content):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.info(f"Saved {filename}")

# Check if content is already saved using a hash
def is_content_saved(content):
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    with content_hashes_lock:
        if content_hash in content_hashes:
            return True
        content_hashes.add(content_hash)
        return False

# Extract and clean HTML content
async def extract_content(session, url):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all('p')
                content = '\n\n'.join([p.get_text() for p in paragraphs])
                return clean_text(content)
        except Exception as e:
            logging.error(f"Failed to retrieve {url}: {e}")
            return None

# Extract and clean PDF content
async def extract_pdf_content(session, url):
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                pdf_content = await response.read()
                with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                    content = ''.join(page.extract_text() + '\n' for page in pdf.pages if page.extract_text())
                    return clean_text(content)
        except Exception as e:
            logging.error(f"Failed to retrieve PDF {url}: {e}")
            return None

# Scrape and clean content from a URL
async def scrape_and_clean(session, url, max_depth, max_files, chunk_size, file_counter):
    global visited_urls
    urls_to_visit = queue.Queue()
    urls_to_visit.put((url, 0))
    base_url = "{0.scheme}://{0.netloc}".format(urlparse(url))

    while not urls_to_visit.empty() and file_counter[0] <= max_files:
        current_url, depth = urls_to_visit.get()
        with visited_urls_lock:
            if current_url in visited_urls or depth > max_depth:
                continue
            visited_urls.add(current_url)

        logging.info(f"Visiting: {current_url} (depth {depth})")

        # Extract content
        content = await (extract_pdf_content(session, current_url) if current_url.endswith('.pdf') else extract_content(session, current_url))

        if content and not is_content_saved(content):
            # Save content in chunks
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                filename = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(current_url.encode()).hexdigest()}_chunk_{file_counter[0]}.md"
                save_to_markdown(filename, chunk)
                file_counter[0] += 1

        # Extract and queue new links
        try:
            soup = BeautifulSoup(content, 'html.parser')
            links = soup.find_all('a')
            for link in links:
                href = link.get('href')
                if href:
                    full_url = normalize_url(urljoin(base_url, href))
                    with visited_urls_lock:
                        if full_url not in visited_urls and full_url.startswith(base_url):
                            urls_to_visit.put((full_url, depth + 1))
                            logging.info(f"Queued: {full_url}")
        except Exception as e:
            logging.error(f"Error processing links from {current_url}: {e}")

        # Random rate limit between requests
        await asyncio.sleep(random.uniform(1, 3))

    logging.info("Scraping completed.")

# Main async function to run multiple scraping tasks concurrently
async def main(start_url, max_depth, max_files, chunk_size, num_threads):
    async with ClientSession() as session:
        file_counter = [1]  # Mutable counter shared across threads
        tasks = [scrape_and_clean(session, start_url, max_depth, max_files, chunk_size, file_counter) for _ in range(num_threads)]
        await asyncio.gather(*tasks)

# Entry point
if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Reinforcement_learning"
    max_depth = 4
    max_files = 500
    chunk_size = 5000
    num_threads = 5

    logging.info("Starting scraping process...")
    asyncio.run(main(start_url, max_depth, max_files, chunk_size, num_threads))
