#!/usr/bin/env python3
"""
Simple launcher for Echo web interface
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Now import and run the web app
from webui.app import create_echo_interface

if __name__ == "__main__":
    print("Creating Echo interface...")
    demo = create_echo_interface()
    print("Launching server on http://127.0.0.1:7861")
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False,
        show_error=True,
        quiet=False,
        prevent_thread_lock=False
    )