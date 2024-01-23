from typing import Optional
import requests


def get_url_content(url: str) -> Optional[str]:
    """Fetches the text content from a URL."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching content from URL: {url}")
        print(e)
        return None
