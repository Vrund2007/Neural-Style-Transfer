import os
import sys

# Add NST_Code to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'NST_Code'))

# Set VERCEL env var so app.py uses /tmp for uploads
os.environ.setdefault('VERCEL', '1')

from app import app

# Vercel expects the WSGI app to be named 'app' or 'handler'
handler = app
