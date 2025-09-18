import subprocess
import sys
import os

# Name of your Streamlit script
streamlit_script = "streamlit_test.py"

# Optional: get the absolute path
script_path = os.path.abspath(streamlit_script)

# Run the Streamlit command
subprocess.run([sys.executable, "-m", "streamlit", "run", script_path])