import os

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