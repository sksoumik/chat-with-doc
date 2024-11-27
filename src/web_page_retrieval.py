"""Web page content retrieval utilities."""
import logging
from typing import Optional
from urllib.parse import urlparse

import requests
from requests.exceptions import RequestException

from .logger import setup_logger
from .models import Document

logger = setup_logger(__name__)


class WebPageRetriever:
    """Handles web page content retrieval operations."""

    @staticmethod
    def is_valid_url(url: str) -> bool:
        """
        Validate if the given string is a valid URL.

        Args:
            url (str): URL to validate

        Returns:
            bool: True if valid URL, False otherwise
        """
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except ValueError:
            return False

    def get_content(self, url: str) -> Optional[Document]:
        """
        Fetch content from a URL and return it as a Document.

        Args:
            url (str): URL to fetch content from

        Returns:
            Optional[Document]: Document containing the web page content,
                              or None if retrieval fails
        """
        if not self.is_valid_url(url):
            logger.error(f"Invalid URL format: {url}")
            return None

        try:
            logger.info(f"Fetching content from URL: {url}")
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            return Document(content=response.text, source=url)

        except RequestException as e:
            logger.error(f"Error fetching content from URL {url}: {str(e)}")
            return None
