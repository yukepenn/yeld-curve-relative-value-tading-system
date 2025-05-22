# yield_curve_rv_strategy/src/data/data_cleaner.py
import pandas as pd
import numpy as np
import sys
import os # Added for BASE_DIR and if __name__ path manipulations

# --- BEGIN PYTHON PATH MODIFICATION FOR DIRECT EXECUTION ---
# This block should be at the top, before any project-specific imports.
if __name__ == '__main__' and __package__ is None:
    # If run directly, __file__ is src/data/data_cleaner.py
    # Want to add 'yield_curve_rv_strategy' (parent of 'src') to path
    # so `from src.utils...` works.
    _project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if _project_root_path not in sys.path:
        sys.path.append(_project_root_path)
# --- END PYTHON PATH MODIFICATION ---

from src.utils.app_logger import get_logger, setup_logging
# No direct config dependency anticipated unless for default cleaning_params

logger = get_logger(__name__)
# BASE_DIR should point to yield_curve_rv_strategy/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def clean_data(df: pd.DataFrame, cleaning_params: dict = None) -> pd.DataFrame:
    if df is None or df.empty:
        logger.warning("Input DataFrame for clean_data is None or empty. Returning as is.")
        return df

    cleaned_df = df.copy()
    logger.info(f"Starting data cleaning. Initial shape: {cleaned_df.shape}")

    if cleaning_params is None:
        cleaning_params = {}

    # Ensure index is DatetimeIndex
    if not isinstance(cleaned_df.index, pd.DatetimeIndex):
        try:
            cleaned_df.index = pd.to_datetime(cleaned_df.index)
            logger.info("Converted index to DatetimeIndex.")
        except Exception as e:
            logger.error(f"Failed to convert index to DatetimeIndex: {e}", exc_info=True)
            return df 

    # 1. Handle Missing Values
    # Convert non-numeric explicitly to numeric, coercing errors to NaN
    for col in cleaned_df.columns:
        # Check if conversion is needed and possible
        if not pd.api.types.is_numeric_dtype(cleaned_df[col]):
            try:
                original_nan_count = cleaned_df[col].isnull().sum()
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                new_nan_count = cleaned_df[col].isnull().sum()
                if new_nan_count > original_nan_count:
                    logger.debug(f"Coerced column '{col}' to numeric. Introduced {new_nan_count - original_nan_count} new NaNs.")
                else:
                    logger.debug(f"Coerced column '{col}' to numeric.")
            except Exception as e: 
                logger.warning(f"Could not convert column '{col}' to numeric: {e}")
    
    initial_nans = cleaned_df.isnull().sum().sum()

    # Interpolation first
    interpolate_method = cleaning_params.get('interpolate_method', None) 
    if interpolate_method:
        logger.debug(f"Interpolating with method: {interpolate_method}")
        # Ensure only numeric columns are interpolated if DataFrame has mixed types
        numeric_cols = cleaned_df.select_dtypes(include=np.number).columns
        if not numeric_cols.empty:
            cleaned_df[numeric_cols] = cleaned_df[numeric_cols].interpolate(method=interpolate_method, axis=0, limit_direction='both')
        else:
            logger.warning("No numeric columns found to interpolate.")

    # Forward fill
    ffill_limit = cleaning_params.get('ffill_limit', None)                                                     
    if ffill_limit is not None : 
        logger.debug(f"Forward-filling NaNs with limit: {ffill_limit if ffill_limit > 0 else 'unlimited'}")
        cleaned_df = cleaned_df.ffill(limit=ffill_limit if ffill_limit > 0 else None)

    # Backward fill
    bfill_limit = cleaning_params.get('bfill_limit', None)
    if bfill_limit is not None:
        logger.debug(f"Backward-filling NaNs with limit: {bfill_limit if bfill_limit > 0 else 'unlimited'}")
        cleaned_df = cleaned_df.bfill(limit=bfill_limit if bfill_limit > 0 else None)
        
    final_nans = cleaned_df.isnull().sum().sum()
    logger.info(f"NaNs handled: {initial_nans} initial -> {final_nans} final.")

    # 2. Outlier Handling
    outlier_std_dev_threshold = cleaning_params.get('outlier_std_dev_threshold', None)
    if outlier_std_dev_threshold and outlier_std_dev_threshold > 0:
        logger.debug(f"Applying outlier clipping at {outlier_std_dev_threshold} std devs.")
        for col in cleaned_df.select_dtypes(include=np.number).columns:
            mean = cleaned_df[col].mean()
            std = cleaned_df[col].std()
            if pd.notna(mean) and pd.notna(std) and std > 1e-6: # Avoid issues with all-NaN or constant columns
                lower_bound = mean - outlier_std_dev_threshold * std
                upper_bound = mean + outlier_std_dev_threshold * std
                original_sum = cleaned_df[col].sum() 
                cleaned_df[col] = cleaned_df[col].clip(lower=lower_bound, upper=upper_bound)
                if cleaned_df[col].sum() != original_sum: # Check if actual clipping happened
                     logger.debug(f"Column '{col}' values clipped for outliers.")
            elif std <= 1e-6 and pd.notna(std):
                 logger.debug(f"Column '{col}' has zero or very low std dev, skipping outlier clipping.")

    # 3. Drop rows if critical columns are still NaN (optional)
    critical_cols = cleaning_params.get('critical_cols_for_dropna', [])
    if critical_cols:
       rows_before_dropna = len(cleaned_df)
       cleaned_df.dropna(subset=critical_cols, inplace=True)
       rows_after_dropna = len(cleaned_df)
       if rows_before_dropna > rows_after_dropna:
           logger.info(f"Dropped {rows_before_dropna - rows_after_dropna} rows with NaNs in critical columns: {critical_cols}. New shape: {cleaned_df.shape}")

    logger.info(f"Data cleaning finished. Final shape: {cleaned_df.shape}")
    return cleaned_df


def merge_data(data_dict: dict[str, pd.DataFrame | None], merge_how: str = 'outer', 
               resample_freq: str = 'B') -> pd.DataFrame | None:
    if not data_dict:
        logger.warning("Data dictionary for merge_data is empty. Returning None.")
        return None

    processed_dfs = []
    df_names_in_order = [] # Keep track of names for suffixing if needed

    for name, df_input in data_dict.items():
        if df_input is None or df_input.empty:
            logger.warning(f"DataFrame for '{name}' is None or empty, skipping in merge.")
            continue
        
        df = df_input.copy()
        
        if not isinstance(df.index, pd.DatetimeIndex):
            try:
                df.index = pd.to_datetime(df.index) 
                logger.debug(f"Converted index of '{name}' to DatetimeIndex.")
            except Exception as e:
                logger.error(f"Failed to convert index of '{name}' to DatetimeIndex: {e}. Skipping.", exc_info=True)
                continue
        
        # Rename columns for clarity before merge, if they are not already specific enough.
        # This is crucial if multiple DataFrames might have same-named columns like 'Close'.
        # Example: if df_input is from Yahoo with 'Close', rename to f'{name}_Close'.
        # The prompt example has df3_zn_f already with prefixed columns.
        # If a df has a generic 'value' column (often from FRED series_to_frame), rename it to `name`.
        if 'value' in df.columns and len(df.columns) == 1:
            df = df.rename(columns={'value': name})
            logger.debug(f"Renamed 'value' column in source '{name}' to '{name}'.")
        elif len(df.columns) == 1 and df.columns[0] != name: # If single column and its name is not the key
             original_col_name = df.columns[0]
             df = df.rename(columns={original_col_name: name})
             logger.debug(f"Renamed column '{original_col_name}' in source '{name}' to '{name}'.")
        # For multi-column DFs (like Yahoo OHLCV), we assume columns are already distinct enough
        # or were made distinct before calling merge_data (e.g., 'ZN_F_Close').
        # The pd.merge suffixes handle remaining clashes.

        processed_dfs.append(df)
        df_names_in_order.append(name)


    if not processed_dfs:
        logger.error("No valid DataFrames to merge.")
        return None

    logger.info(f"Merging {len(processed_dfs)} DataFrames using '{merge_how}' strategy.")
    
    merged_df = processed_dfs[0]
    if len(processed_dfs) > 1:
        for i in range(1, len(processed_dfs)):
            # Suffixes: if column names clash, apply suffix based on the incoming df's original name
            # However, pandas merge suffixes are applied universally if there's a clash.
            # A more robust way is to ensure unique column names *before* merge if specific naming is desired.
            # For now, the default suffixing of pandas is fine.
            merged_df = pd.merge(merged_df, processed_dfs[i], left_index=True, right_index=True, how=merge_how,
                                 suffixes=(f'_{df_names_in_order[i-1]}', f'_{df_names_in_order[i]}'))


    if resample_freq and isinstance(merged_df.index, pd.DatetimeIndex): 
        logger.info(f"Resampling merged DataFrame to frequency: {resample_freq}")
        merged_df = merged_df.resample(resample_freq).asfreq()
    elif resample_freq and not isinstance(merged_df.index, pd.DatetimeIndex):
        logger.warning(f"Cannot resample merged DataFrame: index is not DatetimeIndex.")


    logger.info(f"Merged DataFrame shape: {merged_df.shape}")
    return merged_df


def align_dates_and_ffill(df: pd.DataFrame, ffill_limit: int = 2) -> pd.DataFrame | None:
    if df is None or df.empty:
        logger.warning("Input DataFrame for align_dates_and_ffill is None or empty.")
        return df
    
    aligned_df = df.copy()
    logger.info(f"Aligning dates and forward-filling. Initial shape: {aligned_df.shape}. Ffill limit: {ffill_limit}")

    if not isinstance(aligned_df.index, pd.DatetimeIndex):
        try:
            aligned_df.index = pd.to_datetime(aligned_df.index)
            logger.info("Converted index to DatetimeIndex for alignment.")
        except Exception as e:
            logger.error(f"Failed to ensure DatetimeIndex for alignment: {e}", exc_info=True)
            return df 

    if ffill_limit is not None: 
        limit_val = ffill_limit if ffill_limit > 0 else None
        aligned_df = aligned_df.ffill(limit=limit_val)
        logger.info(f"Forward-filled data. Shape after ffill: {aligned_df.shape}")
    
    return aligned_df


# Example Usage
if __name__ == '__main__':
    # Setup logging for testing
    # BASE_DIR is defined at module level.
    # The setup_logging function in app_logger uses its own BASE_DIR to resolve relative log_file paths.
    test_log_file_path_in_logs_dir = os.path.join("logs", "data_cleaner_test.log")
    setup_logging(log_level_str="DEBUG", log_file=test_log_file_path_in_logs_dir)

    logger.info("--- Running data_cleaner.py tests ---")

    # Create Sample Data for testing
    dates1 = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05'])
    data1 = {'value': [1, np.nan, 3, 4, 5.5]} # Using 'value' to test renaming
    df1_seriesA = pd.DataFrame(data1, index=dates1)

    dates2 = pd.to_datetime(['2023-01-02', '2023-01-03', '2023-01-04', '2023-01-06', '2023-01-07'])
    data2 = {'seriesB_val': [10, 11, np.nan, 13, 14]} # Using specific name
    df2_seriesB = pd.DataFrame(data2, index=dates2)
    
    dates3 = pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'])
    data3 = {
        'Open': [100, 101, 102, 103], 'High': [105, 106, 103, 104],
        'Low': [99, 100, 101, 102], 'Close': [101, 102, 101.5, 103.5],
        'Volume': [1000, 1200, 1100, 1300]
    }
    df3_zn_f = pd.DataFrame(data3, index=dates3)
    # Prefix columns to simulate how multi-column data might be prepared
    df3_zn_f.columns = [f"ZN_F_{col}" for col in df3_zn_f.columns]


    # Test 1: Merge Data
    logger.info("--- Test 1: Merging data_dict ---")
    data_to_merge = {
        "seriesA": df1_seriesA, 
        "seriesB": df2_seriesB, # df2_seriesB has 'seriesB_val' which should become 'seriesB'
        "ZN_F_ohlcv": df3_zn_f 
    }
    # merge_data renames 'seriesB_val' to 'seriesB' because len(cols)==1 and colname != key
    # merge_data renames 'value' in df1_seriesA to 'seriesA'
    merged = merge_data(data_to_merge, merge_how='outer', resample_freq='B') 
    logger.info(f"Merged data (resampled to 'B'):\n{merged}")
    
    # Test 2: Align Dates and Ffill
    if merged is not None:
        logger.info("--- Test 2: Aligning dates and ffilling merged data ---")
        aligned = align_dates_and_ffill(merged, ffill_limit=1) 
        logger.info(f"Aligned and ffilled data:\n{aligned}")
    else:
        logger.error("Merged data is None, skipping Test 2.")
        aligned = None 

    # Test 3: Clean Data
    if aligned is not None:
        logger.info("--- Test 3: Cleaning aligned data ---")
        cleaning_parameters = {
            'interpolate_method': 'linear', 
            'ffill_limit': 1, # Fill up to 1 day after interpolation
            'bfill_limit': 1, # Fill up to 1 day after ffill
            'outlier_std_dev_threshold': 2.0,
            'critical_cols_for_dropna': ['seriesA'] # Example: drop row if seriesA is still NaN
        }
        cleaned = clean_data(aligned, cleaning_params=cleaning_parameters)
        logger.info(f"Cleaned data:\n{cleaned}")
        
        logger.info("--- Test 3.1: Cleaning data with more NaNs to show interpolation ---")
        df_for_interp = pd.DataFrame({'val': [1, np.nan, np.nan, np.nan, 5]}, 
                                     index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))
        cleaned_interp = clean_data(df_for_interp, {'interpolate_method':'linear', 'ffill_limit':0}) # ffill_limit 0 for unlimited
        logger.info(f"Interpolated data (linear then unlimited ffill):\n{cleaned_interp}")

    else:
        logger.error("Aligned data is None, skipping Test 3.")

    logger.info("--- data_cleaner.py tests complete ---")
