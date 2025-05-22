from fredapi import Fred
import pandas as pd
import os
import sys # For sys.path modification

# --- BEGIN PYTHON PATH MODIFICATION FOR DIRECT EXECUTION ---
# This block should be at the top, before any project-specific imports,
# to ensure `utils.app_logger` and `utils.config_loader` can be found when script is run directly.
_current_script_path = os.path.abspath(__file__) # .../src/data/fetch_fred.py
_src_directory = os.path.dirname(os.path.dirname(_current_script_path)) # .../src/
if _src_directory not in sys.path:
    sys.path.insert(0, _src_directory)
# --- END PYTHON PATH MODIFICATION ---

try:
    # Preferred import if script is run as part of a package
    from ..utils.app_logger import get_logger, setup_logging
    from ..utils.config_loader import load_config
except ImportError:
    # Fallback for direct execution (e.g., `python src/data/fetch_fred.py`)
    # Requires `src` to be in sys.path (handled by the block above).
    print("Attempting fallback import for app_logger and config_loader (utils.*) due to ImportError.")
    from utils.app_logger import get_logger, setup_logging
    from utils.config_loader import load_config

logger = get_logger(__name__)

# Assuming this file is yield_curve_rv_strategy/src/data/fetch_fred.py
# BASE_DIR should point to yield_curve_rv_strategy/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Attempt to load config to get a default API key
DEFAULT_CONFIG = {}
DEFAULT_FRED_API_KEY = "YOUR_FRED_API_KEY" # Default placeholder
try:
    DEFAULT_CONFIG = load_config() # Loads from default path: config/config.yaml
    if DEFAULT_CONFIG and "data_settings" in DEFAULT_CONFIG and "fred_api_key" in DEFAULT_CONFIG["data_settings"]:
        DEFAULT_FRED_API_KEY = DEFAULT_CONFIG["data_settings"]["fred_api_key"]
    else:
        logger.warning("FRED API key not found under 'data_settings.fred_api_key' in default config.")
except Exception as e:
    logger.warning(f"Could not load default config for FRED API key: {e}. API key must be provided directly or set in config.")


def fetch_series(series_id: str, start_date: str, end_date: str, 
                 api_key: str = None, data_raw_path: str = None) -> pd.DataFrame | None:
    
    resolved_api_key = api_key if api_key else DEFAULT_FRED_API_KEY
    if not resolved_api_key or resolved_api_key == "YOUR_FRED_API_KEY":
        logger.error(f"FRED API key not provided or is placeholder for series {series_id}. Cannot fetch data.")
        return None

    logger.info(f"Fetching FRED series {series_id} from {start_date} to {end_date}")
    try:
        fred = Fred(api_key=resolved_api_key)
        series_data = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
        
        if series_data.empty:
            logger.warning(f"No data found for FRED series {series_id} for the given date range.")
            return None 

        df = series_data.to_frame(name=series_id) 
        logger.info(f"Successfully fetched {len(df)} data points for FRED series {series_id}")

        if data_raw_path:
            resolved_save_path_dir: str
            if not os.path.isabs(data_raw_path):
                resolved_save_path_dir = os.path.join(BASE_DIR, data_raw_path)
            else:
                resolved_save_path_dir = data_raw_path
            
            if not os.path.exists(resolved_save_path_dir):
                try:
                    os.makedirs(resolved_save_path_dir, exist_ok=True)
                    logger.info(f"Created directory: {resolved_save_path_dir}")
                except OSError as e:
                    logger.error(f"Could not create directory {resolved_save_path_dir}: {e}. Data for {series_id} will not be saved.")
                    data_raw_path = None # Prevent further save attempts for this call

            if data_raw_path: # Check again
                filename = f"{series_id}_fred.csv"
                full_file_path = os.path.join(resolved_save_path_dir, filename)
                
                try:
                    df.to_csv(full_file_path)
                    logger.info(f"Saved data for {series_id} to {full_file_path}")
                except IOError as e:
                    logger.error(f"Could not save data for {series_id} to {full_file_path}: {e}")
        
        return df

    except Exception as e:
        logger.error(f"Error fetching data for FRED series {series_id}: {e}", exc_info=True)
        return None


def fetch_multiple_series(series_ids: list[str], start_date: str, end_date: str, 
                          api_key: str = None, data_raw_path: str = None) -> dict[str, pd.DataFrame | None]:
    data_dict: dict[str, pd.DataFrame | None] = {}
    resolved_api_key = api_key if api_key else DEFAULT_FRED_API_KEY 

    if not resolved_api_key or resolved_api_key == "YOUR_FRED_API_KEY":
        logger.error("FRED API key not provided or is placeholder. Cannot fetch multiple series.")
        return {series_id: None for series_id in series_ids}

    for series_id in series_ids:
        df = fetch_series(series_id, start_date, end_date, api_key=resolved_api_key, data_raw_path=data_raw_path)
        data_dict[series_id] = df
    logger.info(f"Finished fetching multiple FRED series. Requested: {len(series_ids)}, Successfully fetched (non-None): {sum(1 for df in data_dict.values() if df is not None)}")
    return data_dict


if __name__ == '__main__':
    # Setup logging for testing this module directly
    # Log file will be created in yield_curve_rv_strategy/logs/
    test_log_file = os.path.join("logs", "fetch_fred_test.log") 
    setup_logging(log_level_str="DEBUG", log_file=test_log_file)

    logger.info("--- Running fetch_fred.py tests ---")
    
    # The DEFAULT_FRED_API_KEY is loaded at module start.
    # Test functions will use this key if no specific key is passed.
    if DEFAULT_FRED_API_KEY == "YOUR_FRED_API_KEY":
        logger.warning("FRED API key is the placeholder 'YOUR_FRED_API_KEY'.")
        logger.warning("Actual data fetching tests will be skipped or will fail.")
        logger.warning("Please set a valid FRED_API_KEY in your config/config.yaml or provide one directly for testing.")
        # To enforce this for automated tests, one might uncomment the following:
        # sys.exit("Exiting tests: Valid FRED API key required for fetch_fred.py tests.")

    test_start_date = "2023-01-01"
    test_end_date = "2023-06-01" 
    # This path is relative to BASE_DIR (yield_curve_rv_strategy/)
    # So files will be saved in yield_curve_rv_strategy/data/raw/
    test_data_raw_path = os.path.join("data", "raw")

    # Test 1: Fetch single series (DGS10)
    series1_id = "DGS10"
    logger.info(f"--- Test 1: Fetching single series '{series1_id}' ---")
    dgs10_data = fetch_series(series1_id, test_start_date, test_end_date, data_raw_path=test_data_raw_path)
    if dgs10_data is not None:
        if not dgs10_data.empty:
            logger.info(f"Successfully fetched {series1_id} data. Shape: {dgs10_data.shape}. Columns: {dgs10_data.columns.tolist()}")
        else: # Should be None if empty, based on current logic
            logger.warning(f"Fetching {series1_id} returned an empty DataFrame (should be None if no data).")
    else:
        logger.error(f"Failed to fetch {series1_id} data (returned None).")


    # Test 2: Fetch another single series (GDP) - different date range
    series2_id = "GDP"
    logger.info(f"--- Test 2: Fetching single series '{series2_id}' (longer history) ---")
    gdp_data = fetch_series(series2_id, "2020-01-01", test_end_date, data_raw_path=test_data_raw_path)
    if gdp_data is not None:
        if not gdp_data.empty:
            logger.info(f"Successfully fetched {series2_id} data. Shape: {gdp_data.shape}. Columns: {gdp_data.columns.tolist()}")
        else:
            logger.warning(f"Fetching {series2_id} returned an empty DataFrame.")
    else:
        logger.error(f"Failed to fetch {series2_id} data (returned None).")

    # Test 3: Fetch multiple series, including an invalid one
    test_series_list = ["DGS2", "DFII20", "INVALIDFREDSERIES"]
    logger.info(f"--- Test 3: Fetching multiple series: {test_series_list} ---")
    multiple_data_results = fetch_multiple_series(test_series_list, test_start_date, test_end_date, data_raw_path=test_data_raw_path)
    
    for symbol, df_result in multiple_data_results.items():
        if df_result is not None:
            if not df_result.empty:
                logger.info(f"Data for {symbol} (Test 3) fetched. Shape: {df_result.shape}. Columns: {df_result.columns.tolist()}")
            else:
                 logger.warning(f"No data returned for {symbol} (Test 3) (empty DataFrame).")
        else: # df_result is None
             logger.warning(f"Fetching data for {symbol} (Test 3) returned None (expected for INVALIDFREDSERIES or if API key issue).")
            
    logger.info("--- fetch_fred.py tests complete ---")
