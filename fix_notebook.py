#!/usr/bin/env python3
"""
Script to fix pandas Series slicing issues in the performance.ipynb notebook.
The issue is that x[1:] and y[1:] on pandas Series causes multi-dimensional indexing errors.
We need to convert to numpy arrays first: x.values[1:] and y.values[1:]
"""

import json
import re

def fix_notebook(input_file, output_file):
    """Fix the notebook by replacing pandas Series slicing with numpy array slicing."""
    
    # Read the notebook
    with open(input_file, 'r') as f:
        notebook = json.load(f)
    
    # Pattern to find plt.plot calls with pandas Series slicing
    pattern = r'plt\.plot\(x\[1:\], y\[1:\]'
    replacement = 'plt.plot(x.values[1:], y.values[1:]'
    
    # Fix all cells
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            # Fix the source code
            if 'source' in cell:
                if isinstance(cell['source'], list):
                    for i, line in enumerate(cell['source']):
                        cell['source'][i] = re.sub(pattern, replacement, line)
                else:
                    cell['source'] = re.sub(pattern, replacement, cell['source'])
    
    # Write the fixed notebook
    with open(output_file, 'w') as f:
        json.dump(notebook, f, indent=1)
    
    print(f"Fixed notebook saved to {output_file}")

if __name__ == "__main__":
    fix_notebook("performance.ipynb", "performance_fixed.ipynb")
