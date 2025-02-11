"""
Discourse Analysis Package

This package provides tools for analyzing discourse patterns in transcripts.
It includes a base analyzer class and specific analyzers for different discourse markers.
"""

from .analyzer import DiscourseAnalyzer
from .visualizer import DiscourseVisualizer
from .markers.right_analyzer import RightAnalyzer

__all__ = ['DiscourseAnalyzer', 'DiscourseVisualizer', 'RightAnalyzer'] 