import json
import os

notebook_path = '/home/datascientist/workspace/github-collabs/ds_projects_collabs/1-cluster_retail_uci/notebook/cluster_retail.ipynb'

def patch_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r') as f:
        try:
            nb = json.load(f)
        except Exception as e:
            print(f"Error loading JSON: {e}")
            return

    cells = nb.get('cells', [])
    patched_count = 0

    for cell in cells:
        if cell['cell_type'] != 'code':
            continue
            
        source = cell.get('source', [])
        new_source = []
        modified = False
        
        for line in source:
            # We want to remove or fix any lingering "from src import..."
            # Valid imports are "from src.features..." etc, IF src is a package
            # But we switched to adding src to path and doing "from features..."
            # Wait, cluster_retail DOES add src to path. 
            # So "from src import functions" is wrong. "from features import cleaning" is right.
            
            stripped = line.strip()
            
            # Legacy patterns to fix
            if 'from src import functions' in stripped:
                new_source.append('from features import cleaning as functions\n')
                modified = True
            elif 'from src import pipeline' in stripped:
                new_source.append('from features import pipeline\n')
                modified = True
            elif 'from src import k_means_function' in stripped:
                new_source.append('from models import training as k_means_function\n')
                modified = True
            elif 'from src.plot_save import' in stripped:
                new_source.append('from visualization.reporting import save_plot_to_folder\n')
                modified = True
            elif 'from src.elbow_method import' in stripped:
                new_source.append('from models.training import plot_elbow_method\n')
                modified = True
            # General "from src import" wildcard catcher if it's not one of the above valid paths (path adjustment makes 'src' not a root package usually unless running from root)
            # Actually, `sys.path.append(.../src)` makes `import features` valid.
            # `import src.features` might fail if `src` folder lacks __init__.py or if path isn't root.
            # Let's trust specific replacements.
            else:
                new_source.append(line)
        
        if modified:
            cell['source'] = new_source
            patched_count += 1

    if patched_count > 0:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {patched_count} cells in {notebook_path}")
    else:
        print("No remaining legacy imports found to patch.")

if __name__ == "__main__":
    patch_notebook()
