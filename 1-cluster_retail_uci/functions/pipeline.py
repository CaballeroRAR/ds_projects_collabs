import pandas as pd
import sys
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
path_to_functions = os.path.join(project_root, '1-cluster_retail_uci', 'functions')
sys.path.append(path_to_functions)

from functions import (
    normalize_column_names,
    df_column_to_string,
    filter_rows_starting_with,
    remove_and_display_unique_prefixes,
    get_abnormal_values,
    display_rows_by_list,
    filter_consecutive_digits,
    exclude_values_by_list,
    drop_na_duplicates_and_zeroes,
)

def cleaning_pipeline(df):
    # 1. Normalize column names
    df = normalize_column_names(df)

    # 2. Column 'invoice' to string
    df = df_column_to_string(df, "invoice", show_head=False)

    # 3 Filter out rows where 'invoice' starts with 'C'
    df_cancellation_invoices = filter_rows_starting_with(df, "invoice", "C")

    # 4. Remove and display unique prefixes in 'invoice' column
    remove_and_display_unique_prefixes(df, "invoice")

    # 5 Column 'stockcode' to string
    df_column_to_string(df, 'stockcode')

    # 6. Remove abnormal codes
    stockcodes_abnormal, count_abnormal = get_abnormal_values(df, 'stockcode', print_list=True)

    # 7 Filter out abnormal invoices ( exclude NOT 6 consecutive digits)
    df, mask, entries_dropped = filter_consecutive_digits(df, 'invoice', 6) 
    print("Entries dropped:", entries_dropped)

    # 8 Drop abnormal stockcodes by list
    df = exclude_values_by_list(df, "stockcode", stockcodes_abnormal)

    # 9. Drop NA, duplicates, zero price
    df = drop_na_duplicates_and_zeroes(df)
    df.describe()
    return df
