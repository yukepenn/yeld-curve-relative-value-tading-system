import yaml
import os
from functools import reduce # For get_config_value
import logging

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Determine the base directory of the project.
# This assumes config_loader.py is in src/utils/
# So, BASE_DIR will be the 'yield_curve_rv_strategy' directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# BASE_DIR should be the project root: 'yield_curve_rv_strategy'
# SCRIPT_DIR = <path_to_project>/yield_curve_rv_strategy/src/utils
# os.path.dirname(SCRIPT_DIR) = <path_to_project>/yield_curve_rv_strategy/src
# os.path.dirname(os.path.dirname(SCRIPT_DIR)) = <path_to_project>/yield_curve_rv_strategy
BASE_DIR = os.path.dirname(os.path.dirname(SCRIPT_DIR))

def _resolve_path(relative_path):
    """Helper function to resolve paths relative to the project's base directory."""
    return os.path.join(BASE_DIR, relative_path)

def load_config(config_path="config/config.yaml"):
    """
    Loads the main configuration file.
    Args:
        config_path (str): Relative path to the config.yaml file from project root.
    Returns:
        dict: Parsed configuration content.
    Raises:
        FileNotFoundError: If the configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    resolved_path = _resolve_path(config_path)
    try:
        with open(resolved_path, 'r') as stream:
            config_data = yaml.safe_load(stream)
            logging.info(f"Configuration file '{resolved_path}' loaded successfully.")
            return config_data
    except FileNotFoundError:
        logging.error(f"Configuration file not found at '{resolved_path}'.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{resolved_path}': {e}")
        raise

def load_curves_config(curves_path="config/curves.yaml"):
    """
    Loads the curves configuration file.
    Args:
        curves_path (str): Relative path to the curves.yaml file from project root.
    Returns:
        dict: Parsed curves configuration content.
    Raises:
        FileNotFoundError: If the curves configuration file is not found.
        yaml.YAMLError: If there's an error parsing the YAML file.
    """
    resolved_path = _resolve_path(curves_path)
    try:
        with open(resolved_path, 'r') as stream:
            curves_data = yaml.safe_load(stream)
            logging.info(f"Curves configuration file '{resolved_path}' loaded successfully.")
            return curves_data
    except FileNotFoundError:
        logging.error(f"Curves configuration file not found at '{resolved_path}'.")
        raise
    except yaml.YAMLError as e:
        logging.error(f"Error parsing YAML file '{resolved_path}': {e}")
        raise

def get_spread_details(spread_name, curves_config=None):
    """
    Retrieves configuration details for a specific spread.
    Args:
        spread_name (str): The name of the spread (e.g., "2s10s").
        curves_config (dict, optional): Pre-loaded curves configuration. 
                                        If None, load_curves_config() is called.
    Returns:
        dict: Configuration for the specified spread, or None if not found.
    """
    if curves_config is None:
        try:
            curves_config = load_curves_config()
        except FileNotFoundError:
            logging.warning("Curves config file not found when trying to get spread details.")
            return None # Or re-raise if strict error handling is preferred

    if curves_config and "spreads" in curves_config and spread_name in curves_config["spreads"]:
        logging.debug(f"Details for spread '{spread_name}' retrieved successfully.")
        return curves_config["spreads"][spread_name]
    else:
        logging.warning(f"Spread '{spread_name}' not found in curves configuration.")
        return None

def get_config_value(key_path, config=None):
    """
    Retrieves a value from the configuration using a dot-separated key path.
    Args:
        key_path (str): Dot-separated path to the desired key (e.g., "data_settings.fred_api_key").
        config (dict, optional): Pre-loaded main configuration. 
                                 If None, load_config() is called.
    Returns:
        The value at the specified key path, or None if the path is invalid or key not found.
    """
    if config is None:
        try:
            config = load_config()
        except FileNotFoundError:
            logging.warning("Main config file not found when trying to get config value.")
            return None # Or re-raise

    if not config:
        return None

    try:
        # Split the key_path and navigate through the dictionary
        value = reduce(lambda d, key: d.get(key) if d else None, key_path.split('.'), config)
        if value is not None:
            logging.debug(f"Value for key path '{key_path}' retrieved successfully.")
        else:
            logging.debug(f"Key path '{key_path}' not found or leads to a None value in config.")
        return value
    except TypeError: # Handles cases where an intermediate key is not a dictionary
        logging.warning(f"Invalid path or structure for key path '{key_path}' in configuration.")
        return None


# Example usage (optional, for testing within the module)
if __name__ == '__main__':
    # When running this script directly (e.g., python src/utils/config_loader.py from the project root)
    # BASE_DIR will be correctly calculated.
    # The load_config and load_curves_config functions use default paths relative to BASE_DIR.
    
    print(f"Attempting to load configs from project base: {BASE_DIR}")
    # The _resolve_path is an internal helper, but we can check its output for the default paths
    print(f"Expected resolved config.yaml path: {_resolve_path('config/config.yaml')}")
    print(f"Expected resolved curves.yaml path: {_resolve_path('config/curves.yaml')}")

    general_config = None
    curves_data_config = None # Renamed to avoid conflict

    # Test loading with default paths, relying on BASE_DIR and _resolve_path
    try:
        general_config = load_config() 
    except FileNotFoundError:
        # This error message in the example block might be confusing if the paths are actually correct
        # The actual FileNotFoundError with the resolved path will be logged by load_config
        print(f"ERROR: Main config file not found. Check paths and BASE_DIR. Attempted path: {_resolve_path('config/config.yaml')}")
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse main config file: {e}")
    
    try:
        curves_data_config = load_curves_config()
    except FileNotFoundError:
        print(f"ERROR: Curves config file not found. Check paths and BASE_DIR. Attempted path: {_resolve_path('config/curves.yaml')}")
    except yaml.YAMLError as e:
        print(f"ERROR: Could not parse curves config file: {e}")

    if general_config:
        print("\nGeneral Config Loaded:")
        # print(general_config)
        print(f"FRED API Key: {get_config_value('data_settings.fred_api_key', general_config)}")
        print(f"XGBoost LR: {get_config_value('model_settings.xgboost_params.learning_rate', general_config)}")
        print(f"Non-existent value: {get_config_value('data_settings.non_existent_key', general_config)}")
        print(f"Log file path: {get_config_value('logging_settings.log_file', general_config)}")


    if curves_data_config: # Use the renamed variable
        print("\nCurves Config Loaded:")
        # print(curves_data_config)
        spread_2s10s = get_spread_details("2s10s", curves_data_config)
        if spread_2s10s:
            print("\nDetails for '2s10s':")
            print(spread_2s10s)
        
        spread_nonexistent = get_spread_details("invalid_spread", curves_data_config)
        if spread_nonexistent is None:
            print("\nCorrectly handled non-existent spread 'invalid_spread'.")

        spread_5s30s = get_spread_details("5s30s") # Test internal loading
        if spread_5s30s:
            print("\nDetails for '5s30s' (loaded internally):")
            print(spread_5s30s)

    # Test get_config_value with internal loading
    print(f"\nDefault Start Date (loaded internally): {get_config_value('data_settings.default_start_date')}")

    # Test error handling for non-existent config files if they were temporarily renamed/moved for testing
    # print("\nTesting non-existent file load (config):")
    # try:
    #     load_config("config/non_existent_config.yaml")
    # except FileNotFoundError:
    #     print("Successfully caught FileNotFoundError for non_existent_config.yaml")
    
    # print("\nTesting non-existent file load (curves):")
    # try:
    #     load_curves_config("config/non_existent_curves.yaml")
    # except FileNotFoundError:
    #     print("Successfully caught FileNotFoundError for non_existent_curves.yaml")
