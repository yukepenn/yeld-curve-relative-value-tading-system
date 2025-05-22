import yfinance as yf
import pandas as pd
import os
from typing import Dict, List, Optional # For type hints

# --- BEGIN PYTHON PATH MODIFICATION FOR DIRECT EXECUTION ---
# This block should be at the top, before any project-specific imports,
# to ensure `utils.app_logger` can be found when script is run directly.
import sys
# Assuming this script is in yield_curve_rv_strategy/src/data/fetch_yahoo.py
# Add yield_curve_rv_strategy/src to sys.path
_current_script_path = os.path.abspath(__file__) # .../src/data/fetch_yahoo.py
_src_directory = os.path.dirname(os.path.dirname(_current_script_path)) # .../src/
if _src_directory not in sys.path:
    sys.path.insert(0, _src_directory)
# --- END PYTHON PATH MODIFICATION ---

# Assuming app_logger.py is in src/utils
# Adjust the import path based on your project structure if necessary
try:
    # This import is preferred if the script is run as part of a package
    # (e.g., `python -m src.data.fetch_yahoo`)
    from ..utils.app_logger import get_logger, setup_logging
except ImportError: 
    # This fallback is for direct execution (e.g., `python src/data/fetch_yahoo.py`)
    # and requires `src` to be in sys.path (handled by the block above).
    print("Attempting fallback import for app_logger (utils.app_logger) due to ImportError.")
    from utils.app_logger import get_logger, setup_logging


# Assuming this file is yield_curve_rv_strategy/src/data/fetch_yahoo.py
# BASE_DIR should point to yield_curve_rv_strategy/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize logger for this module
# We get the logger instance here, but its configuration (handlers, level)
# will be set by setup_logging, typically called from an entry point or main script.
# For the if __name__ == '__main__': block, setup_logging is called explicitly.
logger = get_logger(__name__)


def fetch_data(symbol: str, start_date: str, end_date: str, 
               data_raw_path: Optional[str] = None, 
               api_settings: Optional[Dict] = None) -> Optional[pd.DataFrame]:
    """
    Fetches historical market data for a given symbol from Yahoo Finance.

    Args:
        symbol (str): The ticker symbol to fetch (e.g., "ZN=F").
        start_date (str): Start date string (e.g., "YYYY-MM-DD").
        end_date (str): End date string (e.g., "YYYY-MM-DD").
        data_raw_path (Optional[str]): Path to the directory for saving raw data.
                                       If None, data is not saved.
                                       If relative, assumed to be relative to project root.
        api_settings (Optional[Dict]): Dictionary for API-specific settings (not heavily used by yfinance).

    Returns:
        Optional[pd.DataFrame]: DataFrame with OHLCV data, or None on failure.
    """
    logger.info(f"Fetching Yahoo Finance data for {symbol} from {start_date} to {end_date}")
    
    # yfinance usually handles user agent and retries internally, but api_settings is for extensibility.
    # Example: yf.set_proxy(proxy_server) if api_settings and api_settings.get("proxy")

    try:
        ticker = yf.Ticker(symbol)
        # yfinance's `end` parameter for daily data is typically exclusive.
        # For example, end='2023-02-01' will fetch data up to and including '2023-01-31'.
        # This is standard for many financial data APIs using daily intervals.
        df = ticker.history(start=start_date, end=end_date)

        if df.empty:
            logger.warning(f"No data found for symbol {symbol} for the period {start_date} to {end_date}.")
            return None
        
        # Standardize column names to upper case for consistency, if needed (yfinance usually returns them capitalized)
        # df.columns = [col.capitalize() for col in df.columns]
        
        logger.info(f"Successfully fetched {len(df)} data points for {symbol}.")

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
                    logger.error(f"Could not create directory {resolved_save_path_dir}: {e}. Data will not be saved.")
                    data_raw_path = None # Prevent further save attempts

            if data_raw_path: # Check again in case it was disabled
                # Sanitize symbol name for filename (more robustly)
                filename_symbol = "".join(c if c.isalnum() else "_" for c in symbol)
                filename = f"{filename_symbol}_yahoo.csv"
                full_file_path = os.path.join(resolved_save_path_dir, filename)
                
                try:
                    df.to_csv(full_file_path)
                    logger.info(f"Saved data for {symbol} to {full_file_path}")
                except IOError as e:
                    logger.error(f"Could not save data for {symbol} to {full_file_path}: {e}")
        
        return df

    except Exception as e:
        # Catching general exceptions from yfinance (e.g., network issues, invalid symbol format before request)
        logger.error(f"Error fetching or processing data for symbol {symbol}: {e}", exc_info=True)
        return None


def fetch_multiple_symbols(symbols: List[str], start_date: str, end_date: str, 
                           data_raw_path: Optional[str] = None, 
                           api_settings: Optional[Dict] = None) -> Dict[str, Optional[pd.DataFrame]]:
    """
    Fetches historical market data for multiple symbols.

    Args:
        symbols (List[str]): A list of ticker symbols.
        start_date (str): Start date string.
        end_date (str): End date string.
        data_raw_path (Optional[str]): Path to the directory for saving raw data.
        api_settings (Optional[Dict]): Dictionary for API-specific settings.

    Returns:
        Dict[str, Optional[pd.DataFrame]]: Dictionary with symbols as keys and DataFrames as values.
                                           Value is None if fetching failed for a symbol.
    """
    data_dict: Dict[str, Optional[pd.DataFrame]] = {}
    for symbol in symbols:
        df = fetch_data(symbol, start_date, end_date, data_raw_path, api_settings)
        data_dict[symbol] = df
    logger.info(f"Finished fetching data for multiple symbols. Requested: {len(symbols)}, Successfully fetched (non-None): {sum(1 for df in data_dict.values() if df is not None)}")
    return data_dict


# Example Usage
if __name__ == '__main__':
    # Setup logging for testing this module directly
    # Log file will be created in yield_curve_rv_strategy/logs/
    # Note: BASE_DIR is defined at the module level.
    # The setup_logging function in app_logger uses its own BASE_DIR to resolve relative log_file paths.
    test_log_file_path_in_logs_dir = os.path.join("logs", "fetch_yahoo_test.log") 
    setup_logging(log_level_str="DEBUG", log_file=test_log_file_path_in_logs_dir)
    
    # Logger is already initialized at module level by the get_logger(__name__) call at the top.
    # No need to re-get it unless for a different name.

    logger.info("--- Running fetch_yahoo.py tests ---")

    # Define a raw data path for testing, relative to project root
    # This will result in files being saved to yield_curve_rv_strategy/data/raw/
    test_data_raw_path = os.path.join("data", "raw")

    test_start_date = "2023-01-01"
    # yfinance end_date is exclusive for daily data, so this fetches up to 2023-01-31
    test_end_date = "2023-02-01" 

    # Test 1: Fetch single common stock symbol
    logger.info("--- Test 1: Fetching single stock symbol (AAPL) ---")
    aapl_data = fetch_data("AAPL", test_start_date, test_end_date, data_raw_path=test_data_raw_path)
    if aapl_data is not None and not aapl_data.empty:
        logger.info(f"Successfully fetched AAPL data. Shape: {aapl_data.shape}. Columns: {aapl_data.columns.tolist()}")
        # print(aapl_data.head())
    else:
        logger.error("Failed to fetch AAPL data or no data returned for Test 1.")

    # Test 2: Fetch a futures symbol
    logger.info("--- Test 2: Fetching future symbol (ZN=F) ---")
    # Note: Futures data availability can be spotty for short/past ranges on Yahoo Finance.
    # Using a slightly wider range or more recent dates might yield better test results if needed.
    zn_data = fetch_data("ZN=F", test_start_date, test_end_date, data_raw_path=test_data_raw_path)
    if zn_data is not None and not zn_data.empty:
        logger.info(f"Successfully fetched ZN=F data. Shape: {zn_data.shape}. Columns: {zn_data.columns.tolist()}")
        # print(zn_data.head())
    else:
        logger.warning(f"Failed to fetch ZN=F data or no data returned for Test 2. This might be due to test date range or market hours for free Yahoo data.")

    # Test 3: Fetch multiple symbols, including an invalid one
    logger.info("--- Test 3: Fetching multiple symbols (ES=F, NQ=F, INVALIDTICKERXYZ) ---")
    symbols_list = ["ES=F", "NQ=F", "INVALIDTICKERXYZ"]
    multiple_data_results = fetch_multiple_symbols(symbols_list, test_start_date, test_end_date, data_raw_path=test_data_raw_path)
    
    for symbol, df_result in multiple_data_results.items():
        if df_result is not None and not df_result.empty:
            logger.info(f"Data for {symbol} (Test 3) fetched. Shape: {df_result.shape}. Columns: {df_result.columns.tolist()}")
        elif df_result is None:
             logger.warning(f"Fetching data for {symbol} (Test 3) returned None (expected for INVALIDTICKERXYZ if it's truly invalid).")
        else: # Empty DataFrame
            logger.warning(f"No data returned for {symbol} (Test 3) (empty DataFrame).")
            
    logger.info("--- fetch_yahoo.py tests complete ---")
