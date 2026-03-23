"""Launch the Streamlit GUI for anomaly detection."""
import subprocess
import sys
import os

# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Run streamlit
subprocess.run([
    sys.executable, 
    "-m", 
    "streamlit", 
    "run", 
    os.path.join(script_dir, "utils", "gui.py")
], cwd=script_dir)
