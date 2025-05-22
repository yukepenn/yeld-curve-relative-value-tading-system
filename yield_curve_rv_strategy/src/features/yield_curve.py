# yield_curve_rv_strategy/src/features/yield_curve.py
import pandas as pd
import numpy as np
import sys
import os

# Correctly add project root to sys.path for direct execution
if __name__ == '__main__' and __package__ is None:
    project_root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    if project_root_path not in sys.path:
        sys.path.append(project_root_path)

from src.utils.app_logger import get_logger, setup_logging

logger = get_logger(__name__)

DEFAULT_ROLLING_WINDOWS = [5, 20, 60]
DEFAULT_LAG_DAYS = [1, 2, 3, 5]

def compute_spread(front_yield_series: pd.Series, back_yield_series: pd.Series) -> pd.Series:
    if not all(isinstance(s, pd.Series) for s in [front_yield_series, back_yield_series]):
        logger.error("Inputs to compute_spread must be pandas Series.")
        return pd.Series(dtype=np.float64) # Return empty series
    if front_yield_series.empty or back_yield_series.empty:
        logger.warning("One or both input series for spread computation are empty.")
        return pd.Series(dtype=np.float64)

    try:
        # Ensure DatetimeIndex (should ideally be handled upstream or checked robustly)
        if not isinstance(front_yield_series.index, pd.DatetimeIndex):
            front_yield_series.index = pd.to_datetime(front_yield_series.index)
        if not isinstance(back_yield_series.index, pd.DatetimeIndex):
            back_yield_series.index = pd.to_datetime(back_yield_series.index)
            
        aligned_front, aligned_back = front_yield_series.align(back_yield_series, join='inner')
        if aligned_front.empty or aligned_back.empty:
            logger.warning("Spread computation resulted in empty series after alignment (no common dates).")
            return pd.Series(dtype=np.float64)
            
        spread = aligned_back - aligned_front
        spread.name = "yield_spread" # Default name
        logger.info(f"Computed spread. Resulting series length: {len(spread)}")
        return spread
    except Exception as e:
        logger.error(f"Error computing spread: {e}", exc_info=True)
        return pd.Series(dtype=np.float64)


def generate_yield_curve_features(spread_series: pd.Series, 
                                  window_sizes: list[int] = None,
                                  lag_days: list[int] = None) -> pd.DataFrame:
    if not isinstance(spread_series, pd.Series):
        logger.error("Input to generate_yield_curve_features must be a pandas Series.")
        return pd.DataFrame()
    if spread_series.empty:
        logger.warning("Input spread_series is empty. Cannot generate features.")
        return pd.DataFrame()

    features_df = pd.DataFrame(index=spread_series.index)
    # Use a copy of the spread to avoid modifying the original Series if it's part of features_df
    current_spread_col = spread_series.copy()
    current_spread_col.name = 'current_spread' # Ensure a specific name if spread_series name is None or generic
    features_df[current_spread_col.name] = current_spread_col

    effective_window_sizes = window_sizes if window_sizes is not None else DEFAULT_ROLLING_WINDOWS
    effective_lag_days = lag_days if lag_days is not None else DEFAULT_LAG_DAYS
    
    logger.info(f"Generating yield curve features. Windows: {effective_window_sizes}, Lags: {effective_lag_days}")

    for lag in effective_lag_days:
        if lag <= 0: 
            logger.warning(f"Lag days must be positive, skipping lag: {lag}")
            continue
        features_df[f'spread_lag_{lag}d'] = current_spread_col.shift(lag)

    for window in effective_window_sizes:
        if window <= 0:
            logger.warning(f"Window size must be positive, skipping window: {window}")
            continue
        
        min_p = max(1, int(window * 0.5)) # Ensure min_periods is reasonable

        features_df[f'spread_ma_{window}d'] = current_spread_col.rolling(window=window, min_periods=min_p).mean()
        rolling_std_series = current_spread_col.rolling(window=window, min_periods=min_p).std()
        features_df[f'spread_std_{window}d'] = rolling_std_series
        features_df[f'spread_mom_{window}d'] = current_spread_col.diff(window)
        
        rolling_mean_series = features_df[f'spread_ma_{window}d'] # Use already computed MA
        # Ensure std dev is notna and greater than a small epsilon to avoid division by zero or near-zero
        features_df[f'spread_zscore_{window}d'] = np.where(
            (rolling_std_series.notna()) & (rolling_std_series > 1e-7), 
            (current_spread_col - rolling_mean_series) / rolling_std_series, 
            0.0 # Default to 0 if std is zero, near-zero, or NaN
        )
    
    logger.info(f"Generated features DataFrame with shape: {features_df.shape}")
    return features_df


def generate_target(spread_series: pd.Series, periods_ahead: int = 1) -> pd.Series:
    if not isinstance(spread_series, pd.Series):
        logger.error("Input to generate_target must be a pandas Series.")
        return pd.Series(dtype=np.float64)
    if spread_series.empty:
        logger.warning("Input spread_series is empty. Cannot generate target.")
        return pd.Series(dtype=np.float64)
    if periods_ahead <= 0:
        logger.error("periods_ahead must be positive.")
        return pd.Series(dtype=np.float64)

    logger.info(f"Generating target for {periods_ahead} period(s) ahead.")
    
    future_spread = spread_series.shift(-periods_ahead)
    target_series = future_spread - spread_series
    target_series.name = f'target_spread_change_{periods_ahead}d' # More descriptive name
    
    logger.info(f"Generated target series. Length: {len(target_series)}, Non-NaN: {target_series.count()}")
    return target_series

# Example Usage
if __name__ == '__main__':
    # Assuming this script is in yield_curve_rv_strategy/src/features/
    # project_root is yield_curve_rv_strategy/
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log_dir = os.path.join(project_root, "logs")
    if not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True)
    
    # The setup_logging function from app_logger uses its own BASE_DIR logic
    # to resolve relative paths for log_file from project root.
    setup_logging(log_level_str="DEBUG", log_file=os.path.join("logs", "yield_curve_features_test.log"))

    logger.info("--- Running yield_curve.py tests ---")

    dates = pd.date_range(start='2023-01-01', periods=100, freq='B') # Business days
    front_yield_data = np.random.rand(100) * 2 + 1.0 # Yields between 1.0 and 3.0
    back_yield_data = front_yield_data + np.random.rand(100) * 1.0 + 0.2 # Ensure back > front mostly
    
    front_yield = pd.Series(front_yield_data, index=dates, name="DGS2")
    back_yield = pd.Series(back_yield_data, index=dates, name="DGS10")

    logger.info("Test 1: Computing spread...")
    spread = compute_spread(front_yield, back_yield)
    assert spread is not None and not spread.empty, "Spread computation failed or returned empty."
    logger.info(f"Spread computed. Head:\n{spread.head()}")

    logger.info("\nTest 2: Generating yield curve features...")
    yc_features = generate_yield_curve_features(spread, window_sizes=[5, 10], lag_days=[1,3])
    assert yc_features is not None and not yc_features.empty, "Feature generation failed or returned empty."
    logger.info(f"YC features generated. Shape: {yc_features.shape}. Head:\n{yc_features.head()}")
    # logger.info(f"NaNs in features:\n{yc_features.isnull().sum()}")


    logger.info("\nTest 3: Generating target variable...")
    target = generate_target(spread, periods_ahead=1)
    assert target is not None and not target.empty, "Target generation failed or returned empty."
    logger.info(f"Target generated. Shape: {target.shape}. Tail:\n{target.tail()}")
    # logger.info(f"NaNs in target: {target.isnull().sum()}")


    logger.info("\nTest 4: Compute spread with misaligned series...")
    dates_short = pd.date_range(start='2023-01-10', periods=50, freq='B')
    front_yield_short = pd.Series(np.random.rand(50) * 2, index=dates_short, name="DGS2_short")
    spread_misaligned = compute_spread(front_yield_short, back_yield)
    assert spread_misaligned is not None and not spread_misaligned.empty and len(spread_misaligned) <= 50, \
        f"Misaligned spread computation issue. Length: {len(spread_misaligned)}"
    logger.info(f"Spread (misaligned) computed. Length: {len(spread_misaligned)}.")

    logger.info("\nTest 5: Empty series handling")
    empty_s = pd.Series(dtype=np.float64)
    assert compute_spread(empty_s, back_yield).empty
    assert compute_spread(front_yield, empty_s).empty
    assert generate_yield_curve_features(empty_s).empty
    assert generate_target(empty_s).empty
    logger.info("Empty series handling tests passed.")
    
    logger.info("--- yield_curve.py tests complete ---")
