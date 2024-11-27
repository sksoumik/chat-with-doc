"""Debug configuration settings."""
import logging
from typing import Any, Dict

DEBUG_CONFIG: Dict[str, Any] = {
    "log_level": logging.DEBUG,
    "enable_streamlit_debug": True,
    "show_error_details": True,
    "development_mode": True,
}
