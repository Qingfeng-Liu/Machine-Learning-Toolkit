import os
import sys
from pathlib import Path
from datetime import datetime


def generate_directory_structure(root_path, max_depth=3, output_file=sys.stdout):
    """
    Generate a visual directory tree structure

    Args:
        root_path (str/Path): Starting directory path
        max_depth (int): Maximum folder depth to display
        output_file: File object or stream for output
    """
    root_path = str(root_path)  # Ensure string path for replacement

    for current_dir, subdirs, files in os.walk(root_path):
        # Skip hidden directories and virtual envs
        subdirs[:] = [d for d in subdirs if not d.startswith('.') and d not in ['venv', '__pycache__']]

        # Calculate relative depth
        rel_path = os.path.relpath(current_dir, root_path)
        depth = 0 if rel_path == '.' else len(rel_path.split(os.sep))

        if depth > max_depth:
            continue

        indent = '    ' * depth
        print(f"{indent}{os.path.basename(current_dir)}/", file=output_file)

        file_indent = '    ' * (depth + 1)
        for file in files:
            if not file.startswith('.'):  # Skip hidden files
                file_marker = ' 🐍' if file.endswith('.py') else ''
                print(f"{file_indent}{file}{file_marker}", file=output_file)


def write_to_readme(readme_path, structure_header="## Project Structure"):
    """Write directory structure to README.md"""
    with open(readme_path, 'a', encoding='utf-8') as f:
        print(f"\n{structure_header}", file=f)
        print("```", file=f)
        generate_directory_structure(Path(readme_path).parent, output_file=f)
        print("```", file=f)
        print(f"\n> Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", file=f)


if __name__ == "__main__":
    # Configure paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    readme_path = project_root / "README.md"

    # Write to README
    try:
        write_to_readme(readme_path)
        print(f"✓ Directory structure appended to {readme_path}")
    except Exception as e:
        print(f"✗ Failed to update README: {str(e)}", file=sys.stderr)

    # Console output
    print("\nLive directory structure:")
    generate_directory_structure(project_root)
