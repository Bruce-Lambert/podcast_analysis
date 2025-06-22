"""
Transcript Processing Package

This package provides tools for processing and combining different types of podcast transcripts.
It handles VTT format transcripts, text transcripts with speaker attribution, and their combination.
"""

from .parsers.vtt_parser import VTTParser
from .parsers.text_parser import TextParser
from .combiners.transcript_combiner import TranscriptCombiner

__all__ = ['VTTParser', 'TextParser', 'TranscriptCombiner']
