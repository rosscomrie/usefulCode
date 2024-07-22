import json
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

def verify_file_integrity(
    input_file: str,
    output_file: str,
    load_func: Callable[[str], Any] = json.load,
    validate_func: Callable[[Any, Any], bool] = lambda x, y: True
) -> bool:
    """
    Verify the integrity of two files by comparing their lengths and optionally validating their contents.

    Parameters
    ----------
    input_file : str
        Path to the input file.
    output_file : str
        Path to the output file.
    load_func : Callable[[str], Any], optional
        Function to load the file contents. Defaults to json.load.
    validate_func : Callable[[Any, Any], bool], optional
        Function to validate individual entries. Defaults to always return True.

    Returns
    -------
    bool
        True if integrity check passes, False otherwise.
    """
    logger.info(f"Verifying integrity of files: {input_file} and {output_file}")

    try:
        with open(input_file, 'r') as f:
            input_data = load_func(f)
        with open(output_file, 'r') as f:
            output_data = load_func(f)
    except Exception as e:
        logger.error(f"Error loading files: {str(e)}")
        return False

    if not isinstance(input_data, list) or not isinstance(output_data, list):
        logger.error("Both input and output data must be lists")
        return False

    if len(input_data) != len(output_data):
        logger.error(
            "Mismatch in number of entries between input and output data",
        )
        return False

    for i, (input_entry, output_entry) in enumerate(zip(input_data, output_data, strict=True)):
        if not validate_func(input_entry, output_entry):
            logger.error(f"Integrity check failed for entry {i}")
            return False

    logger.info("Integrity verification completed successfully")
    return True