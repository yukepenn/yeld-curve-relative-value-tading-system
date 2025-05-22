import pandas as pd
import numpy as np
from typing import Union

def calculate_sharpe_ratio(returns_series: pd.Series, risk_free_rate: float, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized Sharpe ratio.

    Args:
        returns_series (pd.Series): A pandas Series of periodic returns (e.g., daily returns).
        risk_free_rate (float): The annualized risk-free rate.
        periods_per_year (int): Number of periods in a year (e.g., 252 for daily, 12 for monthly).

    Returns:
        float: The annualized Sharpe ratio.
    """
    if not isinstance(returns_series, pd.Series):
        raise TypeError("returns_series must be a pandas Series.")
    if not isinstance(risk_free_rate, (int, float)):
        raise TypeError("risk_free_rate must be a numeric value.")
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")

    if returns_series.empty:
        return 0.0  # Or np.nan, depends on how you want to handle empty series

    # Calculate periodic risk-free rate
    periodic_risk_free_rate = risk_free_rate / periods_per_year
    
    # Calculate excess returns
    excess_returns = returns_series - periodic_risk_free_rate
    
    mean_excess_return = excess_returns.mean()
    std_dev_excess_return = excess_returns.std()

    if std_dev_excess_return == 0 or np.isnan(std_dev_excess_return):
        # Handle cases:
        # 1. No volatility: If mean excess return is positive, Sharpe is infinite (or very large).
        #    If mean is zero or negative, Sharpe is typically 0 or negative.
        #    Returning 0.0 is a common simplification for zero std dev.
        # 2. NaN std_dev (e.g. from single data point): return NaN or 0.
        if mean_excess_return > 0:
            return np.inf # Or a large number, or 0 if preferred for flat returns
        elif mean_excess_return <= 0:
            return 0.0 
        else: # handles np.isnan(mean_excess_return)
            return np.nan


    sharpe_ratio = (mean_excess_return / std_dev_excess_return) * np.sqrt(periods_per_year)
    return sharpe_ratio

def calculate_max_drawdown(pnl_series: pd.Series) -> float:
    """
    Calculates the maximum drawdown from a P&L or equity curve.

    Args:
        pnl_series (pd.Series): A pandas Series representing the Profit and Loss (P&L) 
                                or equity curve over time. Assumes values are > 0 for percentage calculation.

    Returns:
        float: The maximum drawdown value (e.g., 0.20 for a 20% drawdown).
               Returns as a positive value.
    """
    if not isinstance(pnl_series, pd.Series):
        raise TypeError("pnl_series must be a pandas Series.")
    if pnl_series.empty:
        return 0.0
    if not (pnl_series > 0).all():
        # If series contains non-positive values, percentage drawdown can be problematic.
        # Consider falling back to absolute drawdown or raising an error.
        # For this implementation, we proceed, but it might lead to inf/-inf if pnl_series hits 0 or negative.
        # A common approach for equity curves is to ensure they start > 0 and stay > 0.
        # If pnl_series can be 0 or negative, absolute drawdown calculation is more robust.
        pass # Allow, but be mindful of potential division by zero if values are not positive.

    cumulative_max = pnl_series.cummax()
    
    # Ensure cumulative_max is not zero to avoid division by zero for percentage calculation.
    # If cumulative_max hits zero (e.g. if pnl_series contains zeros), this can lead to 'inf' or 'nan'.
    # Replace zero in cumulative_max with a very small number or handle appropriately.
    # For simplicity here, we assume pnl_series and thus cummax remain positive if percentage is used.
    # If pnl_series can represent actual P&L (not just equity curve), it can be negative.
    # The problem states "P&L or equity curve". Equity curves are typically positive.
    
    drawdown = (pnl_series - cumulative_max) / cumulative_max
    # drawdown will be <= 0. Max drawdown is the minimum value (largest negative).
    max_drawdown_value = drawdown.min()
    
    if np.isinf(max_drawdown_value):
        # This can happen if cumulative_max was 0 at some point.
        # Fallback to calculating absolute drawdown if percentage is problematic.
        # For now, we return it as is, or one might choose to return a large number or np.nan
        # A truly robust solution here depends on specific requirements for pnl_series contents.
        # Let's assume for an equity curve, it should remain positive.
        # If it's P&L that can go to 0 or negative, absolute drawdown is better:
        # absolute_drawdown = cumulative_max - pnl_series
        # return absolute_drawdown.max()
        pass

    return abs(max_drawdown_value) if pd.notna(max_drawdown_value) else 0.0


def annualize_return(periodic_return_mean: float, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized return from the mean of periodic returns.

    Args:
        periodic_return_mean (float): The mean of periodic returns (e.g., mean daily return).
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: The annualized return.
    """
    if not isinstance(periodic_return_mean, (int, float)):
        raise TypeError("periodic_return_mean must be a numeric value.")
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")
        
    return periodic_return_mean * periods_per_year

def annualize_volatility(periodic_return_std: float, periods_per_year: int = 252) -> float:
    """
    Calculates the annualized volatility from the standard deviation of periodic returns.

    Args:
        periodic_return_std (float): The standard deviation of periodic returns (e.g., std of daily returns).
        periods_per_year (int): Number of periods in a year.

    Returns:
        float: The annualized volatility.
    """
    if not isinstance(periodic_return_std, (int, float)):
        raise TypeError("periodic_return_std must be a numeric value.")
    if not isinstance(periods_per_year, int) or periods_per_year <= 0:
        raise ValueError("periods_per_year must be a positive integer.")

    return periodic_return_std * np.sqrt(periods_per_year)

def bp_to_decimal(basis_points: Union[float, int]) -> float:
    """
    Converts a value in basis points to a decimal.

    Args:
        basis_points (Union[float, int]): The value in basis points (e.g., 50 bp).

    Returns:
        float: The decimal value (e.g., 0.0050).
    """
    if not isinstance(basis_points, (int, float)):
        raise TypeError("basis_points must be a numeric value.")
    return basis_points / 10000.0

def decimal_to_bp(decimal_value: float) -> float:
    """
    Converts a decimal value to basis points.

    Args:
        decimal_value (float): The decimal value (e.g., 0.0050).

    Returns:
        float: The basis points value (e.g., 50 bp).
    """
    if not isinstance(decimal_value, (int, float)): # Also allow int for convenience, e.g. 0
        raise TypeError("decimal_value must be a numeric value.")
    return decimal_value * 10000.0

# Example usage (optional, for testing)
if __name__ == '__main__':
    print("--- Math Utils Tests ---")
    
    # Test Data
    equity_values = [100, 102, 101, 103, 102, 105, 103, 106, 106, 105] # Added some flat/down moves
    equity = pd.Series(equity_values, name="Equity")
    returns = equity.pct_change().dropna()
    
    print(f"\nEquity Curve:\n{equity}")
    print(f"\nPeriodic Returns:\n{returns}")

    # --- Sharpe Ratio ---
    print("\n--- Sharpe Ratio Tests ---")
    rfr_annual = 0.01 # 1% annualized risk-free rate
    
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate=rfr_annual, periods_per_year=252)
    print(f"Calculated Sharpe Ratio (annualized): {sharpe:.4f}")

    # Test Sharpe with flat returns (std_dev_excess_return will be 0)
    flat_returns = pd.Series([0.0001, 0.0001, 0.0001, 0.0001]) # Small positive mean
    sharpe_flat_positive_mean = calculate_sharpe_ratio(flat_returns, rfr_annual, 252)
    print(f"Sharpe (flat positive returns, vs rfr): {sharpe_flat_positive_mean}") # Expect inf if mean > rfr_periodic, else 0

    flat_returns_at_rfr = pd.Series([rfr_annual/252] * 5)
    sharpe_flat_at_rfr = calculate_sharpe_ratio(flat_returns_at_rfr, rfr_annual, 252)
    print(f"Sharpe (flat returns at rfr): {sharpe_flat_at_rfr}") # Expect 0.0

    flat_returns_below_rfr = pd.Series([0.00001] * 5) # below typical rfr
    sharpe_flat_below_rfr = calculate_sharpe_ratio(flat_returns_below_rfr, rfr_annual, 252)
    print(f"Sharpe (flat returns below rfr): {sharpe_flat_below_rfr}") # Expect 0.0 (or neg inf if handled differently)
    
    empty_returns = pd.Series([], dtype=float)
    sharpe_empty = calculate_sharpe_ratio(empty_returns, rfr_annual)
    print(f"Sharpe (empty returns): {sharpe_empty}") # Expect 0.0

    single_return = pd.Series([0.01])
    sharpe_single = calculate_sharpe_ratio(single_return, rfr_annual) # Std will be NaN
    print(f"Sharpe (single return): {sharpe_single}") # Expect NaN

    # --- Max Drawdown ---
    print("\n--- Max Drawdown Tests ---")
    # Using equity for pnl_series for max drawdown calculation
    max_dd = calculate_max_drawdown(equity) 
    print(f"Calculated Max Drawdown (percentage): {max_dd:.4%}")

    # Test with a series that goes to zero or negative (if we were to change the function to support it)
    # equity_with_zero = pd.Series([100, 50, 0, 50])
    # max_dd_abs = calculate_max_drawdown_absolute(equity_with_zero) -> needs a different function
    # print(f"Max Drawdown (absolute, with zero): {max_dd_abs}")

    # Test with an all-increasing series (drawdown should be 0)
    increasing_equity = pd.Series([100, 101, 102, 103])
    max_dd_increasing = calculate_max_drawdown(increasing_equity)
    print(f"Max Drawdown (increasing series): {max_dd_increasing:.4%}") # Expect 0.0000%

    empty_series = pd.Series([], dtype=float)
    max_dd_empty = calculate_max_drawdown(empty_series)
    print(f"Max Drawdown (empty series): {max_dd_empty:.4%}") # Expect 0.0000%

    # --- Annualize Return ---
    print("\n--- Annualize Return Tests ---")
    mean_daily_return = returns.mean()
    ann_ret = annualize_return(mean_daily_return, periods_per_year=252)
    print(f"Mean Daily Return: {mean_daily_return:.6f}")
    print(f"Annualized Return: {ann_ret:.4%}")

    # --- Annualize Volatility ---
    print("\n--- Annualize Volatility Tests ---")
    std_daily_return = returns.std()
    ann_vol = annualize_volatility(std_daily_return, periods_per_year=252)
    print(f"Std Dev Daily Return: {std_daily_return:.6f}")
    print(f"Annualized Volatility: {ann_vol:.4%}")

    # --- Basis Point Conversions ---
    print("\n--- Basis Point Conversion Tests ---")
    bp_val = 50
    dec_val = bp_to_decimal(bp_val)
    print(f"{bp_val} bp to decimal: {dec_val}") # Expect 0.005
    
    reverted_bp = decimal_to_bp(dec_val)
    print(f"{dec_val} decimal to bp: {reverted_bp}") # Expect 50.0

    bp_val_float = 25.5
    dec_val_float = bp_to_decimal(bp_val_float)
    print(f"{bp_val_float} bp to decimal: {dec_val_float}") # Expect 0.00255

    reverted_bp_float = decimal_to_bp(dec_val_float)
    print(f"{dec_val_float} decimal to bp: {reverted_bp_float}") # Expect 25.5

    print("\n--- End of Tests ---")
