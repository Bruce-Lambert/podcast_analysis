"""
Transcript Processing Package

This package provides tools for processing and combining different types of transcripts.
It includes parsers for VTT and text formats, and a combiner to merge them with speaker attribution.
"""

from .vtt_parser import VTTParser
from .text_parser import TextParser
from .transcript_combiner import TranscriptCombiner

__all__ = ['VTTParser', 'TextParser', 'TranscriptCombiner'] 