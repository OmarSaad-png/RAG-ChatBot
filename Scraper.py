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
from urllib.parse import urljoin, urldefrag, urlparse
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential

# Initialize logging
logging.basicConfig(filename='scraper.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Thread-safe data structures
visited_urls = set()
visited_urls_lock = asyncio.Lock()

content_hashes = set()
content_hashes_lock = asyncio.Lock()

# Async semaphore for rate limiting
semaphore = asyncio.Semaphore(5)  # Limit concurrent requests


def normalize_url(url):
    """Normalize URLs to avoid duplicates."""
    url, _ = urldefrag(url)  # Remove fragment
    parsed_url = urlparse(url)
    return f"{parsed_url.scheme}://{parsed_url.netloc}{parsed_url.path}"


def clean_text(text):
    """Clean extracted text."""
    text = ' '.join(text.split())
    return text.encode('ascii', 'ignore').decode('ascii')


def save_to_markdown(filename, content):
    """Save content to a Markdown file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(content)
    logging.info(f"Saved {filename}")


async def is_content_saved(content):
    """Check if content is already saved using a hash."""
    content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()
    async with content_hashes_lock:
        if content_hash in content_hashes:
            return True
        content_hashes.add(content_hash)
        return False


async def extract_content(session, url):
    """Extract and clean HTML content."""
    async with semaphore:
        try:
            async with session.get(url, timeout=10) as response:
                response.raise_for_status()
                html = await response.text()

                # Parse and extract content using BeautifulSoup
                soup = BeautifulSoup(html, 'html.parser')
                paragraphs = soup.find_all(['p', 'div'])
                content = "\n".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
                return clean_text(content) if content else None

        except Exception as e:
            logging.error(f"Failed to retrieve {url}: {e}")
            return None


async def extract_pdf_content(session, url):
    """Extract and clean PDF content."""
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


async def scrape_and_clean(session, queue, max_depth, chunk_size, file_counter):
    """Scrape, extract links, and manage recursion."""
    global visited_urls

    while not queue.empty():
        current_url, depth = await queue.get()

        async with visited_urls_lock:
            if current_url in visited_urls or depth > max_depth:
                continue
            visited_urls.add(current_url)

        logging.info(f"Visiting: {current_url} (depth {depth})")

        # Extract content
        content = await (extract_pdf_content(session, current_url) if current_url.endswith('.pdf') else extract_content(session, current_url))

        if content and not await is_content_saved(content):
            # Save content in chunks
            for i in range(0, len(content), chunk_size):
                chunk = content[i:i + chunk_size]
                filename = f"output/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(current_url.encode()).hexdigest()}_chunk_{file_counter[0]}.md"
                save_to_markdown(filename, chunk)
                file_counter[0] += 1

        # Extract and queue new links only for HTML pages
        if content and not current_url.endswith('.pdf'):
            try:
                soup = BeautifulSoup(content, 'html.parser')  # Parse content to extract links
                links = soup.find_all('a', href=True)
                base_url = "{0.scheme}://{0.netloc}".format(urlparse(current_url))

                for link in links:
                    href = link.get('href')
                    if href and not href.startswith('#'):
                        full_url = normalize_url(urljoin(base_url, href))

                        async with visited_urls_lock:
                            if full_url not in visited_urls and full_url.startswith(base_url):
                                logging.info(f"Queueing: {full_url} (depth {depth + 1})")
                                await queue.put((full_url, depth + 1))

            except Exception as e:
                logging.error(f"Error processing links from {current_url}: {e}")


async def main(start_url, max_depth, max_files, chunk_size, num_threads):
    """Main async function to run multiple scraping tasks concurrently."""
    queue = asyncio.Queue()
    await queue.put((start_url, 0))

    async with ClientSession() as session:
        file_counter = [1]
        tasks = [scrape_and_clean(session, queue, max_depth, chunk_size, file_counter) for _ in range(num_threads)]
        await asyncio.gather(*tasks)


if __name__ == "__main__":
    start_url = "https://en.wikipedia.org/wiki/Reinforcement_learning"
    max_depth = 2  # Set to desired depth
    max_files = 5000
    chunk_size = 5000
    num_threads = 5

    logging.info("Starting scraping process...")
    asyncio.run(main(start_url, max_depth, max_files, chunk_size, num_threads))
