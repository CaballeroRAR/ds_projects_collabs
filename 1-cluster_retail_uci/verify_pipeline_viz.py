import json
import os

notebook_path = '/home/datascientist/workspace/github-collabs/ds_projects_collabs/1-cluster_retail_uci/notebook/storytelling_cluster_retail.ipynb'

def patch_notebook():
    if not os.path.exists(notebook_path):
        print(f"Error: Notebook not found at {notebook_path}")
        return

    with open(notebook_path, 'r') as f:
        nb = json.load(f)

    cells = nb.get('cells', [])
    patched_count = 0

    for cell in cells:
        # Find the cleaning cell
        source_joined = "".join(cell.get('source', []))
        if "clean_df = pipeline.cleaning_pipeline(raw_df)" in source_joined:
            # We want to add a markdown cell BEFORE this one if it doesn't exist, 
            # or just rely on the user seeing the code.
            # actually, let's print a confirmation in the code
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "clean_df = pipeline.cleaning_pipeline(raw_df)" in line:
                    new_source.append("\nprint('✅ Executed Project Leader Cleaning Pipeline (Steps 1-9)')\n")
            cell['source'] = new_source
            patched_count += 1

    if patched_count > 0:
        with open(notebook_path, 'w') as f:
            json.dump(nb, f, indent=1)
        print(f"Successfully patched {patched_count} cells in {notebook_path}")
    else:
        print("No matching cells found to patch.")

if __name__ == "__main__":
    patch_notebook()
