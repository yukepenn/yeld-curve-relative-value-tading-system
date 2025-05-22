import logging
import os
# from .config_loader import get_config_value # If fetching from config

# Store the base directory of the project to correctly resolve log file paths
# This file is yield_curve_rv_strategy/src/utils/app_logger.py
# BASE_DIR should point to yield_curve_rv_strategy/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_logger_initialized = False
_active_log_file_path = None # To keep track of the current file handler's path

def setup_logging(log_level_str: str = "INFO", log_file: str = None, config: dict = None):
    global _logger_initialized, _active_log_file_path

    # Config integration placeholder (as in prompt)
    # if config:
    #     log_level_str = get_config_value("logging_settings.log_level", config) or log_level_str
    #     log_file_from_config = get_config_value("logging_settings.log_file", config)
    #     if log_file_from_config:
    #         log_file = log_file_from_config

    numeric_log_level = getattr(logging, log_level_str.upper(), None)
    if not isinstance(numeric_log_level, int):
        # Use basicConfig for this initial warning if logger isn't set up.
        logging.basicConfig(level=logging.WARNING)
        logging.warning(f"Invalid log level: {log_level_str}. Defaulting to INFO.")
        numeric_log_level = logging.INFO
        log_level_str = "INFO"


    logger = logging.getLogger() # Root logger
    logger.setLevel(numeric_log_level) # Set root logger level

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s")

    # Console Handler: Add if no console handler exists, or update existing one's level.
    console_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
            console_handler = h
            break
    
    if console_handler is None:
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(numeric_log_level) # Set/Update level for console


    # File Handler: Add/replace/remove based on log_file parameter
    current_file_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.FileHandler):
            current_file_handler = h
            break # Assuming only one file handler for simplicity, or the first one found

    if log_file: # User wants file logging
        resolved_log_file_path: str
        if not os.path.isabs(log_file):
            resolved_log_file_path = os.path.join(BASE_DIR, log_file)
        else:
            resolved_log_file_path = log_file

        if current_file_handler and current_file_handler.baseFilename != resolved_log_file_path:
            # If existing file handler is for a different file, remove it
            logger.removeHandler(current_file_handler)
            current_file_handler.close()
            current_file_handler = None
            _active_log_file_path = None
        
        if not current_file_handler:
            log_dir = os.path.dirname(resolved_log_file_path)
            if log_dir and not os.path.exists(log_dir):
                try:
                    os.makedirs(log_dir, exist_ok=True)
                except OSError as e:
                    logger.error(f"Could not create log directory '{log_dir}': {e}")
                    # Fallback: disable file logging for this call
                    log_file = None 
            
            if log_file: # Check again, might have been disabled
                fh = logging.FileHandler(resolved_log_file_path, mode='a')
                fh.setFormatter(formatter)
                logger.addHandler(fh)
                current_file_handler = fh # The new or existing (if path matched) handler
                _active_log_file_path = resolved_log_file_path
                if not _logger_initialized or (current_file_handler and current_file_handler.level != numeric_log_level): # Log file change or first init
                     logger.info(f"Logging to file: {resolved_log_file_path} at level {log_level_str.upper()}")


        if current_file_handler: # Set/Update level for the file handler
            current_file_handler.setLevel(numeric_log_level)

    elif current_file_handler: # User wants console-only, but a file handler exists
        logger.removeHandler(current_file_handler)
        current_file_handler.close()
        logger.info(f"Stopped logging to file: {_active_log_file_path}")
        _active_log_file_path = None

    if not _logger_initialized:
        logger.info(f"Logging system initialized. Root logger level: {log_level_str.upper()}.")
        _logger_initialized = True
    else:
        logger.info(f"Logging system (re)configured. Root logger level: {log_level_str.upper()}.")


def get_logger(name: str) -> logging.Logger:
    # Ensures that if get_logger is called before setup_logging, basic logging is available.
    # Best practice is to call setup_logging explicitly at app start.
    if not _logger_initialized:
        # print("Warning: Logging system not explicitly initialized. Setting up with default INFO level to console.")
        setup_logging() 
    return logging.getLogger(name)


# Example Usage (for testing within the module)
if __name__ == '__main__':
    print(f"Project BASE_DIR determined as: {BASE_DIR}")
    
    # --- Test 1: Initial setup (Console only, INFO level) ---
    print("\n--- Test 1: Console Logging (INFO) ---")
    setup_logging(log_level_str="INFO", log_file=None)
    logger_t1 = get_logger("Test1.Console")
    logger_t1.debug("T1 DEBUG: Should NOT appear on console.")
    logger_t1.info("T1 INFO: Should appear on console.")
    logger_t1.warning("T1 WARNING: Should appear on console.")

    # --- Test 2: Adding File logging (DEBUG level) ---
    print("\n--- Test 2: Adding File Logging (DEBUG) & Updating Level ---")
    test_log_file = os.path.join("logs", "app_test.log") # Relative to BASE_DIR
    
    setup_logging(log_level_str="DEBUG", log_file=test_log_file)
    logger_t2 = get_logger("Test2.FileConsole")
    
    logger_t2.debug(f"T2 DEBUG: Should appear on console and in file '{test_log_file}'.")
    logger_t2.info(f"T2 INFO: Should appear on console and in file '{test_log_file}'.")
    
    # Test logger from Test 1 after root level changed to DEBUG
    logger_t1.debug("T1 DEBUG (after T2 setup): Should now appear on console and in file.")
    
    abs_log_path_t2 = os.path.join(BASE_DIR, test_log_file)
    print(f"Check log file for Test 2 messages at: {abs_log_path_t2}")

    # --- Test 3: Re-configuring with same File, different level (WARNING) ---
    print("\n--- Test 3: Re-configuring with same File, different level (WARNING) ---")
    setup_logging(log_level_str="WARNING", log_file=test_log_file) # Same log file
    logger_t3 = get_logger("Test3.Warning")

    logger_t3.debug("T3 DEBUG: Should NOT appear (level is WARNING).")
    logger_t3.info("T3 INFO: Should NOT appear (level is WARNING).")
    logger_t3.warning(f"T3 WARNING: Should appear on console and in file '{test_log_file}'.")
    
    # Test loggers from T1 and T2 after root level changed to WARNING
    logger_t1.info("T1 INFO (after T3 setup): Should NOT appear.")
    logger_t1.warning("T1 WARNING (after T3 setup): Should appear on console and file.")
    logger_t2.debug("T2 DEBUG (after T3 setup): Should NOT appear.")
    logger_t2.warning("T2 WARNING (after T3 setup): Should appear on console and file.")

    # --- Test 4: Switching to a different log file ---
    print("\n--- Test 4: Switching to a new log file (INFO) ---")
    new_log_file = os.path.join("logs", "app_test_new.log")
    setup_logging(log_level_str="INFO", log_file=new_log_file)
    logger_t4 = get_logger("Test4.NewFile")
    logger_t4.debug("T4 DEBUG: Should NOT appear (level is INFO).")
    logger_t4.info(f"T4 INFO: Should appear on console and in new file '{new_log_file}'.")
    logger_t3.warning(f"T3 WARNING (after T4 setup): Should appear on console and in NEW file (since root level allows WARNING).")
    abs_log_path_t4 = os.path.join(BASE_DIR, new_log_file)
    print(f"Check new log file for Test 4 messages at: {abs_log_path_t4}")
    print(f"Old log file '{abs_log_path_t2}' should no longer be receiving new logs.")

    # --- Test 5: Back to console-only logging (ERROR level) ---
    print("\n--- Test 5: Back to console-only logging (ERROR) ---")
    setup_logging(log_level_str="ERROR", log_file=None)
    logger_t5 = get_logger("Test5.ConsoleOnlyError")
    logger_t5.info("T5 INFO: Should NOT appear (level is ERROR).")
    logger_t5.warning("T5 WARNING: Should NOT appear (level is ERROR).")
    logger_t5.error("T5 ERROR: Should appear on console ONLY.")
    logger_t4.info("T4 INFO (after T5 setup): Should NOT appear on console. File logging to new_log_file stopped.")

    # --- Test 6: Invalid log level string ---
    print("\n--- Test 6: Using an invalid log level string (current level ERROR) ---")
    # The warning about invalid level will be logged at WARNING level (default for basicConfig).
    # The effective logging level will remain ERROR because INFO (the fallback) is less restrictive.
    setup_logging(log_level_str="INVALID_LEVEL_XYZ", log_file=None) # Stays console only
    logger_t6 = get_logger("Test6.InvalidLevel")
    # The initial warning "Invalid log level..." should have appeared from basicConfig.
    # The "Logging system (re)configured..." message will be at ERROR level.
    logger_t6.info("T6 INFO: Should NOT appear (level is ERROR).")
    logger_t6.error("T6 ERROR: Should appear on console.")

    # --- Test 7: get_logger before setup_logging (uses default setup) ---
    print("\n--- Test 7: Call get_logger before any explicit setup_logging ---")
    # This requires resetting _logger_initialized, which is tricky in a single script run.
    # The get_logger() has a fallback to call setup_logging() with defaults.
    # To truly test this in isolation, _logger_initialized would need to be False.
    # For now, we'll assume it demonstrates that get_logger() returns a logger.
    # If _logger_initialized is True from previous tests, it just gets a logger.
    # If it were False, get_logger would call setup_logging() which prints "Logging system initialized..."
    logger_t7 = get_logger("Test7.ImplicitSetup")
    logger_t7.info("T7 INFO: Behavior depends on state of _logger_initialized. If re-init, will be INFO to console.")
    logger_t7.error("T7 ERROR: Should appear on console.")


    print("\n--- End of Logging Tests ---")
    if _active_log_file_path:
        print(f"File logging is currently ACTIVE to: {_active_log_file_path}")
    else:
        print(f"File logging is currently INACTIVE.")
