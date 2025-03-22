#!/usr/bin/env python3
"""
Convenience script to run the trading entry point prediction model
"""

import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from main import main, get_args

if __name__ == "__main__":
    args = get_args()
    main(args)
