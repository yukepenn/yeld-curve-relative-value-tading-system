# yield_curve_rv_strategy/src/features/macro_features.py
import pandas as pd
import numpy as np
import sys
import os

if __name__ == '__main__' and __package__ is None:
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)

from src.utils.app_logger import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_ROLLING_WINDOWS_MACRO = [5, 20, 60] # Default for macro features if needed for generic rolling stats

# Example default config if no specific config is passed.
# This would typically come from a more global config or be explicitly defined.
DEFAULT_MACRO_SERIES_CONFIG = {
    # Assumes columns like 'VIX_Close', 'SP500_Close', 'FEDFUNDS' might exist in cleaned_data
    'VIX_Close': {'type': 'level', 'lags': [1, 5, 20]}, # Level and its lags
    'SP500_Close': {'type': 'returns', 'windows': [1, 5, 20]}, # Daily, weekly, monthly (approx) returns
    'FEDFUNDS': {'type': 'level_change', 'periods': [1, 22, 66]} # Daily, monthly, quarterly changes
}


def generate_macro_features(cleaned_data: pd.DataFrame, 
                            macro_series_config: dict = None,
                            default_windows: list[int] = None) -> pd.DataFrame:
    if not isinstance(cleaned_data, pd.DataFrame):
        logger.error("Input cleaned_data must be a pandas DataFrame.")
        return pd.DataFrame()
    if cleaned_data.empty:
        logger.warning("Input cleaned_data is empty. Cannot generate macro features.")
        return pd.DataFrame()
    if not isinstance(cleaned_data.index, pd.DatetimeIndex):
        logger.error("cleaned_data must have a DatetimeIndex.")
        return pd.DataFrame()

    features_df = pd.DataFrame(index=cleaned_data.index)
    
    config_to_use = macro_series_config if macro_series_config is not None else DEFAULT_MACRO_SERIES_CONFIG
    windows_to_use = default_windows if default_windows is not None else DEFAULT_ROLLING_WINDOWS_MACRO
    
    logger.info(f"Generating macro features using config: {config_to_use}")

    for series_col, params in config_to_use.items():
        if series_col not in cleaned_data.columns:
            logger.warning(f"Series '{series_col}' not found in cleaned_data. Skipping.")
            continue

        base_series = cleaned_data[series_col].copy()
        feature_type = params.get('type', 'level') # Default to 'level' if type not specified
        
        logger.debug(f"Processing series: {series_col}, type: {feature_type}")

        if feature_type == 'level':
            # Use the series as is (could be a feature itself)
            features_df[f'{series_col}_level'] = base_series
            # Generate lags if specified
            for lag in params.get('lags', []):
                if lag > 0:
                    features_df[f'{series_col}_lag_{lag}d'] = base_series.shift(lag)
        
        elif feature_type == 'returns':
            # Generate returns for specified windows
            for window in params.get('windows', windows_to_use): # Use specific windows from config or fallback
                if window > 0:
                    features_df[f'{series_col}_ret_{window}d'] = base_series.pct_change(periods=window)
        
        elif feature_type == 'level_change':
            # Generate absolute changes for specified periods
            for period in params.get('periods', []):
                if period > 0:
                    features_df[f'{series_col}_chg_{period}d'] = base_series.diff(periods=period)
        
        # Optionally, add generic rolling stats (mean, std) for any processed base_series or its primary transform
        if params.get('add_rolling_stats', False): # Example: add a flag in config
            for window in windows_to_use: # Use the general default_windows passed to the function
                if window > 0:
                    # Decide what to take rolling stats on, e.g. base_series or a primary transform
                    # For simplicity, let's assume base_series for now
                    min_p = max(1, int(window * 0.5))
                    features_df[f'{series_col}_ma_{window}d'] = base_series.rolling(window=window, min_periods=min_p).mean()
                    features_df[f'{series_col}_std_{window}d'] = base_series.rolling(window=window, min_periods=min_p).std()

    # Drop rows with all NaNs that might have been created by shifts/diffs at the beginning
    features_df.dropna(how='all', inplace=True)
    
    logger.info(f"Generated macro features DataFrame with shape: {features_df.shape}")
    return features_df


# Example Usage
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    # Log file path relative to project root for setup_logging
    setup_logging(log_level_str="DEBUG", log_file=os.path.join("logs", "macro_features_test.log"))

    logger.info("--- Running macro_features.py tests ---")

    # Create sample cleaned_data DataFrame
    dates = pd.date_range(start='2023-01-01', periods=100, freq='B')
    sample_data = {
        'DGS2': np.random.rand(100) * 1 + 1.0, # Example yield
        'DGS10': np.random.rand(100) * 2 + 2.0, # Example yield
        'VIX_Close': np.random.rand(100) * 10 + 15, # VIX values
        'SP500_Close': np.random.rand(100) * 500 + 3800, # SP500 values
        'FEDFUNDS': np.concatenate([np.full(50, 2.5), np.full(50, 2.75)]) # Fed funds rate change mid-series
    }
    cleaned_df_sample = pd.DataFrame(sample_data, index=dates)

    logger.info(f"Sample cleaned_data Head:\n{cleaned_df_sample.head()}")

    # Test 1: Generate features with default config (expecting VIX, SP500, FEDFUNDS to be processed)
    logger.info("\nTest 1: Generating macro features with default config...")
    macro_features_default = generate_macro_features(cleaned_df_sample.copy()) # Pass a copy
    if macro_features_default.empty and not any(col in cleaned_df_sample.columns for col in DEFAULT_MACRO_SERIES_CONFIG.keys()):
        logger.warning("Test 1: No features generated with default config AND no default series found in input.")
    elif macro_features_default.empty and any(col in cleaned_df_sample.columns for col in DEFAULT_MACRO_SERIES_CONFIG.keys()):
        logger.error("Test 1: No features generated with default config, BUT default series WERE present.")
    else:
        logger.info(f"Macro features (default config) generated. Shape: {macro_features_default.shape}")
        logger.info(f"Head:\n{macro_features_default.head()}")
        logger.info(f"Tail:\n{macro_features_default.tail()}")


    # Test 2: Generate features with custom config
    logger.info("\nTest 2: Generating macro features with custom config...")
    custom_config = {
        'VIX_Close': {'type': 'level', 'lags': [1, 3], 'add_rolling_stats': True},
        'SP500_Close': {'type': 'returns', 'windows': [1, 5, 10]}, # Specific windows for returns
        'NonExistentCol': {'type': 'level', 'lags': [1]} # Test missing column handling
    }
    custom_general_windows = [5, 10] # For rolling stats in VIX_Close if add_rolling_stats is true
    
    macro_features_custom = generate_macro_features(cleaned_df_sample.copy(), 
                                                    macro_series_config=custom_config,
                                                    default_windows=custom_general_windows)
    if macro_features_custom.empty:
         logger.warning("Test 2: No features generated with custom config (check logic).")
    else:
        logger.info(f"Macro features (custom config) generated. Shape: {macro_features_custom.shape}")
        logger.info(f"Head:\n{macro_features_custom.head()}")
        # Check if 'NonExistentCol' features were skipped (they should be)
        assert not any('NonExistentCol' in col for col in macro_features_custom.columns), \
            "Features for NonExistentCol should not be present."
        logger.info("Correctly skipped 'NonExistentCol'.")
        # Check if VIX rolling stats are present as per custom_config and custom_general_windows
        assert 'VIX_Close_ma_5d' in macro_features_custom.columns, "VIX_Close_ma_5d missing."
        assert 'VIX_Close_std_10d' in macro_features_custom.columns, "VIX_Close_std_10d missing."
        # Check if SP500 returns are generated with specified windows
        assert 'SP500_Close_ret_1d' in macro_features_custom.columns, "SP500_Close_ret_1d missing."
        assert 'SP500_Close_ret_10d' in macro_features_custom.columns, "SP500_Close_ret_10d missing."


    # Test 3: Empty DataFrame input
    logger.info("\nTest 3: Testing with empty DataFrame input...")
    empty_df_features = generate_macro_features(pd.DataFrame())
    assert empty_df_features.empty, "Empty DataFrame input should result in empty features DataFrame."
    logger.info("Empty DataFrame input handled correctly.")

    # Test 4: DataFrame without DatetimeIndex
    logger.info("\nTest 4: Testing with DataFrame without DatetimeIndex...")
    non_dt_index_df = pd.DataFrame(sample_data) # No index=dates
    non_dt_features = generate_macro_features(non_dt_index_df)
    assert non_dt_features.empty, "DataFrame without DatetimeIndex should result in empty features DataFrame."
    logger.info("DataFrame without DatetimeIndex handled correctly.")

    logger.info("--- macro_features.py tests complete ---")
