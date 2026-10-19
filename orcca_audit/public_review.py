"""Exact approved public summary, containing no original study excerpts."""
from pathlib import Path

PUBLIC_SUMMARY_SHA256 = "56b69c9e4c3b61619d6d8d246cbfb613e6a44d0671dbb61354b3b1dc9d798778"

def bundled_public_review():
    path = Path(__file__).with_name("public_research_summary.json")
    return str(path), PUBLIC_SUMMARY_SHA256
