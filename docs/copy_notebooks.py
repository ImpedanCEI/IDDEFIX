import shutil
import os

# Define source and destination directories
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
source_dir = os.path.join(repo_root, "examples")
dest_dir = os.path.join(os.path.dirname(__file__), "examples")

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# List of notebooks to copy
notebooks = [
    "001_analytical_resonator.ipynb",
    "002_extrapolation_sim_data.ipynb",
    "003_beam_wire_scanner.ipynb",
    "004a_sps_transition_wake.ipynb",
    "004b_sps_transition_impedance.ipynb",
]

# Copy each notebook to the destination
for notebook in notebooks:
    shutil.copy(os.path.join(source_dir, notebook), os.path.join(dest_dir, notebook))
    print(f"Copied {notebook} to {dest_dir}")
