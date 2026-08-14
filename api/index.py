import os
import sys

# Ensure NST_Code is in Python module search path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NST_Code'))

from app import app
