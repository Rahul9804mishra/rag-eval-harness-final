"""
Shared pytest fixtures.
"""
import os
import sys

# Make sure the project root is on sys.path for all tests.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
