import os

# Path to the MONet_scripts folder
scripts_folder = os.path.join(os.path.dirname(__file__), '..', '..','MAIA_scripts')

# Output rst file path
output_rst = os.path.join(os.path.dirname(__file__),'..', 'source', 'apidocs', 'scripts.rst')

# Get all script filenames (without extension)
script_files = [
    os.path.splitext(f)[0]
    for f in os.listdir(scripts_folder)
    if os.path.isfile(os.path.join(scripts_folder, f)) and f.endswith('.py') and f.startswith('MAIA_')
]

# Sort scripts alphabetically
script_files.sort()

# RST content
rst_content = """Scripts
=======================

MAIA Scripts
-----------------------
.. toctree::
   :maxdepth: 1

"""

for script in script_files:
    rst_content += f"   {script}\n"

# Write to output file
with open(output_rst, 'w') as f:
    f.write(rst_content)

# Create individual rst files for each script
scripts_output_dir = os.path.join(os.path.dirname(__file__), '..', 'source', 'apidocs',)
os.makedirs(scripts_output_dir, exist_ok=True)

for script in script_files:
    script_rst_path = os.path.join(scripts_output_dir, f"{script}.rst")
    script_rst_content = f"""{script} script
==============================================================================

.. automodule:: {script}
.. argparse::
   :ref: {script}.get_arg_parser
   :prog: {script}
"""
    with open(script_rst_path, 'w') as script_file:
        script_file.write(script_rst_content)