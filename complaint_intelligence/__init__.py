"""Classical NLP tools for Arabic and French customer complaints."""

from .language import detect_language
from .routing import route_department

__all__ = ["detect_language", "route_department"]

