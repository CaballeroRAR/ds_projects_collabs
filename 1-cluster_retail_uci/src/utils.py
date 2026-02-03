import os
import io
import pandas as pd
import ipywidgets as widgets
from IPython.display import display

def save_plot_to_folder(fig, filename, folder_name="graph_img"):
    """
    Saves a matplotlib figure to a specific folder.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        The figure object to save.
    filename : str
        The name of the file (e.g., 'cluster_plot.png').
    folder_name : str, default='graph_img'
        The directory to save the image in.
    """
    # 1. Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder created: {folder_name}")
    
    # 2. Construct the full file path
    file_path = os.path.join(folder_name, filename)
    
    # 3. Save the figure
    # dpi=300 is high quality; bbox_inches='tight' prevents labels from being cut off
    fig.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"Image saved to: {file_path}")

def load_pkl_to_dataframe(var_name="df"):
    """
    Displays an upload button to select a .pkl file and loads it into a Pandas DataFrame.
    The DataFrame is assigned to a global variable with the specified name.
    
    Parameters:
    -----------
    var_name : str, default='df'
        The name of the global variable to assign the DataFrame to.
    """
    
    upload_widget = widgets.FileUpload(
        accept='.pkl',  # Accept only .pkl files
        multiple=False,  # Single file upload
        description='Upload PKL'
    )
    
    def on_upload_change(change):
        # Access the uploaded file content
        # ipywidgets 8.x structure: change['new'][0] is a dict with 'content', 'name', etc.
        # ipywidgets 7.x structure: change['new'] is a dict where keys are filenames
        
        # We'll try to handle the structure generically or assume 8.x for modern environments, 
        # but the provided code in pipeline.py had a specific logic. 
        # I will replicate the logic from pipeline.py.
        
        new_value = change['new']
        if not new_value:
            return

        try:
             # Try to get content from the list (ipywidgets 8.x)
            if isinstance(new_value, tuple) or isinstance(new_value, list):
                 uploaded_file = new_value[0]
                 content = uploaded_file['content']
                 filename = uploaded_file['name']
            # Try to get content from the specific dict key (ipywidgets 7.x)
            elif isinstance(new_value, dict):
                 # In 7.x, the key is the filename
                 filename = list(new_value.keys())[0]
                 content = new_value[filename]['content']
            else:
                print("Unknown widget return format.")
                return

            print(f"Loaded: {filename}")
            
            # Load into DataFrame
            df = pd.read_pickle(io.BytesIO(content))
            
            # Assign to the function attribute so it can be retrieved
            load_pkl_to_dataframe.df = df
            
            print(f"DataFrame loaded successfully! Shape: {df.shape}")
            display(df.head())

        except Exception as e:
            print(f"Error loading file: {e}")

    upload_widget.observe(on_upload_change, names='value')
    display(upload_widget)
