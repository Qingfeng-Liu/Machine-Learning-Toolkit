import os
import sys
from pathlib import Path
from datetime import datetime


def generate_directory_structure(root_path, max_depth=3, output_file=sys.stdout):
    """
    Generate a sorted visual directory tree structure with file type indicators

    Args:
        root_path (str/Path): Root directory path
        max_depth (int): Maximum folder depth to display (default: 3)
        output_file: Output file object or stream (default: stdout)
    """
    root_path = str(root_path)
    file_icons = {
        '.py': ' 🐍',
        '.ipynb': ' 📓',
        '.pkl': ' 🧠',
        '.pth': ' 🧠',
        '.csv': ' 📊',
        '.json': ' 📝'
    }

    for current_dir, subdirs, files in os.walk(root_path):
        # Filter and sort directories
        subdirs[:] = sorted([
            d for d in subdirs
            if not d.startswith('.')
               and d not in ['venv', '__pycache__', 'node_modules']
        ])

        # Calculate depth
        rel_path = os.path.relpath(current_dir, root_path)
        depth = 0 if rel_path == '.' else len(rel_path.split(os.sep))

        if depth > max_depth:
            continue

        # Print directory
        indent = '    ' * depth
        print(f"{indent}{os.path.basename(current_dir)}/", file=output_file)

        # Print files with icons
        file_indent = '    ' * (depth + 1)
        for file in sorted(files):
            if file.startswith('.'):
                continue

            ext = os.path.splitext(file)[1]
            icon = file_icons.get(ext, ' 📄')
            print(f"{file_indent}{file}{icon}", file=output_file)


def update_readme_structure(readme_path, structure_header="## Project Structure"):
    """
    Update directory structure section in README.md

    Args:
        readme_path (str/Path): Path to README.md
        structure_header (str): Header marking structure section
    """
    try:
        with open(readme_path, 'r+', encoding='utf-8') as f:
            content = f.readlines()
            f.seek(0)

            # Remove existing structure section
            new_content = []
            skip = False
            for line in content:
                if line.strip().startswith(structure_header):
                    skip = True
                elif skip and line.strip().startswith("```"):
                    skip = False
                    continue
                if not skip:
                    new_content.append(line)

            f.truncate()
            f.writelines(new_content)

            # Append new structure
            print(f"\n{structure_header}", file=f)
            print("```", file=f)
            generate_directory_structure(Path(readme_path).parent, output_file=f)
            print("```", file=f)
            print(f"\n> Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)

        print(f"✓ Directory structure updated in {readme_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to update README: {str(e)}", file=sys.stderr)
        return False


if __name__ == "__main__":
    # Configuration
    PROJECT_ROOT = Path(__file__).parent.parent
    README_PATH = PROJECT_ROOT / "README.md"

    # Update README
    success = update_readme_structure(README_PATH)

    # Live preview
    if success:
        print("\nCurrent directory structure:")
        generate_directory_structure(PROJECT_ROOT)
