"""pytest configuration for semantic-cache tests."""
import sys
import pathlib

# Make the package importable from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
