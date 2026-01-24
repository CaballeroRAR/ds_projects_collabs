# AGENTS.md

This file contains guidelines and commands for agentic coding agents working in this repository.

## Project Overview

This is a data science collaborative repository focused on customer segmentation using unsupervised learning. The main project (`1-cluster_retail_uci`) implements a complete data pipeline for cleaning retail transaction data and applying K-Means clustering for RFM (Recency, Frequency, Monetary) analysis.

## Build/Lint/Test Commands

### Python Environment
- **Python Version**: 3.13 (as specified in pyrightconfig.json)
- **Type Checking**: Use `pyright` for static type checking
- **Package Management**: No requirements.txt found - dependencies are imported directly (pandas, numpy, sklearn, matplotlib, seaborn)

### Testing
- **No formal test suite**: Currently no pytest or unittest framework detected
- **Manual Testing**: Use Jupyter notebooks for testing and validation
- **Pipeline Testing**: `src/notebook_pipeline_test.ipynb` contains pipeline integration tests
- **Single Test Execution**: Run specific notebook cells or create isolated test functions

### Development Workflow
```bash
# Type checking
pyright

# Run pipeline test notebook
jupyter notebook src/notebook_pipeline_test.ipynb

# Main analysis notebook
jupyter notebook notebook/cluster_retail.ipynb
```

## Code Style Guidelines

### Import Organization
```python
# Standard library imports first
import sys
import os

# Third-party imports next
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Local imports last (use relative imports within packages)
from .functions import (
    normalize_column_names,
    df_column_to_string,
    # ... other functions
)
```

### Naming Conventions
- **Functions**: `snake_case` (e.g., `normalize_column_names`, `filter_rows_starting_with`)
- **Variables**: `snake_case` (e.g., `df_clean`, `col_name`, `starting_letter`)
- **Constants**: `UPPER_SNAKE_CASE` (not commonly used in this codebase)
- **Classes**: `PascalCase` (no classes defined yet, but follow convention)

### Type Hints
- **Functions**: Include parameter and return type hints in docstrings rather than annotations
- **Parameters**: Use `pandas.DataFrame`, `str`, `bool`, `int`, `list` in docstring type fields
- **Returns**: Specify return types in docstring `Returns` section

### Function Documentation
Follow the established docstring format:
```python
def function_name(param1, param2, optional_param=default):
    """
    Brief description of what the function does.
    
    Parameters:
    -----------
    param1 : type
        Description of parameter
    param2 : type
        Description of parameter
    optional_param : type, optional
        Description of optional parameter (default is value)
    
    Returns:
    --------
    return_type
        Description of what is returned
    """
```

### Error Handling
- **Validation**: Check column existence with `if column_name not in df.columns:`
- **Type Checking**: Use `pd.api.types.is_string_dtype()` for type validation
- **Clear Messages**: Provide descriptive error messages with available alternatives
- **Graceful Failures**: Use `ValueError` for invalid inputs with helpful messages

### Data Processing Patterns
- **Copy Data**: Always use `df.copy()` when creating new DataFrames to avoid SettingWithCopyWarning
- **Chain Operations**: Prefer method chaining for simple transformations
- **Intermediate Variables**: Use descriptive variable names for complex multi-step operations
- **Logging**: Include print statements for operation feedback (e.g., "Rows removed:", "Model fitted.")

### File Structure
```
1-cluster_retail_uci/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # Main cleaning pipeline
│   ├── functions.py         # Utility functions
│   ├── k_means_function.py  # Clustering functions
│   ├── viz_functions.py     # Visualization functions
│   ├── elbow_method.py      # Cluster optimization
│   └── plot_save.py         # Plot utilities
├── notebook/
│   ├── cluster_retail.ipynb  # Main analysis
│   └── df_2010-2011.pkl     # Cleaned data
└── dataset/                 # Raw data files
```

### Pipeline Development
- **Modular Design**: Create small, focused functions for each cleaning step
- **State Management**: Return transformed DataFrames for pipeline chaining
- **Reproducibility**: Use `random_state` parameters for ML models
- **Documentation**: Include step-by-step comments in pipeline functions

### Machine Learning Conventions
- **Scaling**: Use `StandardScaler` from sklearn for feature preprocessing
- **Clustering**: Set `n_init='auto'` in KMeans to avoid future deprecation warnings
- **Random States**: Use consistent random states (e.g., 123, 42) for reproducibility
- **Model Return**: Return both transformed data and fitted model objects

### Jupyter Notebook Best Practices
- **Import Cells**: Keep imports organized at the top of notebooks
- **Path Management**: Use `os.path.join()` and `sys.path.append()` for module imports
- **Data Persistence**: Save cleaned data as pickle files (`.pkl`) for faster loading
- **Visualization**: Use matplotlib/seaborn with consistent figure sizing and styling

### Code Quality
- **No Comments**: Avoid inline code comments unless specifically requested
- **Descriptive Names**: Use self-documenting variable and function names
- **Consistent Patterns**: Follow established patterns from existing codebase
- **Minimal Dependencies**: Use only necessary libraries (pandas, numpy, sklearn, matplotlib, seaborn)

## Special Notes

- **Relative Imports**: Use `from .functions import` within the src package
- **Data Cleaning**: Extensive data cleaning pipeline with regex patterns for validation
- **RFM Analysis**: Focus on recency, frequency, monetary customer segmentation
- **Visualization**: Emphasis on clear, informative plots for business insights
- **No Testing Framework**: Manual testing through notebooks - consider adding pytest for future development