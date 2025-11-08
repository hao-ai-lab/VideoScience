import requests
from bs4 import BeautifulSoup
import sys

def scrape_text_from_url(url):
    """
    Fetches the content of a given URL and extracts all visible text.
    
    This function removes <script> and <style> tags before extracting text
    to avoid including code and CSS in the output.
    """
    try:
        # Set headers to mimic a common web browser. Some websites block
        # requests that don't look like they're from a browser.
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Connection': 'keep-alive'
        }
        
        # Make the HTTP request with a timeout
        response = requests.get(url, headers=headers, timeout=10)
        
        # This will raise an HTTPError if the response was unsuccessful (4xx or 5xx)
        response.raise_for_status()
        
        # Use BeautifulSoup to parse the HTML content
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Find and remove all <script> and <style> elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()  # Remove the tag from the soup
            
        # Get the text.
        text = soup.get_text()
        
        # Clean up the text:
        # 1. Break into lines and remove leading/trailing whitespace from each line
        lines = (line.strip() for line in text.splitlines())
        # 2. Break multi-headlines into a-line-per-phrase
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        # 3. Join the lines, but only if they are not empty
        cleaned_text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return cleaned_text

    except requests.exceptions.HTTPError as errh:
        print(f"Http Error: {errh}", file=sys.stderr)
    except requests.exceptions.ConnectionError as errc:
        print(f"Error Connecting: {errc}", file=sys.stderr)
    except requests.exceptions.Timeout as errt:
        print(f"Timeout Error: {errt}", file=sys.stderr)
    except requests.exceptions.RequestException as err:
        print(f"An unexpected error occurred: {err}", file=sys.stderr)
    
    # Return None if any exception occurred
    return None