import json
import os

notebook_path = '/home/datascientist/workspace/github-collabs/ds_projects_collabs/1-cluster_retail_uci/notebook/cluster_retail.ipynb'

def patch_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    patched_count = 0

    for cell in cells:
        source_joined = "".join(cell.get('source', []))
        
        # Look for the import of save_plot_to_folder and add plot_outlier_density
        if 'from visualization.reporting import save_plot_to_folder' in source_joined:
            new_source = []
            for line in cell['source']:
                if 'from visualization.reporting import save_plot_to_folder' in line:
                    # Update to import everything or specifically the missing function
                    new_source.append("from visualization.reporting import save_plot_to_folder, plot_outlier_density\n")
                else:
                    new_source.append(line)
            cell['source'] = new_source
            patched_count += 1
            print("Patched reporting imports")
            
    if patched_count > 0:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {patched_count} cells in {notebook_path}")
    else:
        print("No matching import cells found to patch.")

if __name__ == "__main__":
    patch_notebook()
