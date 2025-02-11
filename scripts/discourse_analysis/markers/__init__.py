"""
Discourse Markers Package

This package contains analyzers for specific discourse markers.
Each analyzer extends the base DiscourseAnalyzer with marker-specific functionality.
"""

from .right_analyzer import RightAnalyzer

__all__ = ['RightAnalyzer'] 