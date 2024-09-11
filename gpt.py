import os

# Define the root directory (current directory in this case)
root_dir = '.'

# Output file to store combined code
output_file = 'combined_code.py'

# Directory to ignore (your virtual environment)
ignore_dir = 'oanda_bot_env'

# Open the output file in write mode
with open(output_file, 'w') as outfile:
    # Walk through all directories and subdirectories
    for dirpath, dirnames, filenames in os.walk(root_dir):
        # Skip the virtual environment directory
        if ignore_dir in dirnames:
            dirnames.remove(ignore_dir)
        
        for filename in filenames:
            # Only process .py files
            if filename.endswith('.py'):
                file_path = os.path.join(dirpath, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        # Write file name and contents to the combined file
                        outfile.write(f"# Contents of {file_path}\n")
                        outfile.write(infile.read())
                        outfile.write("\n\n")
                except UnicodeDecodeError:
                    print(f"Skipping {file_path} due to encoding issues.")

print(f"All Python files (excluding {ignore_dir}) have been combined into {output_file}")