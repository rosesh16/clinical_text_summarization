import sys
import traceback
from datetime import datetime


def get_error_details(error: Exception, error_detail: sys) -> str:
    """
    Extracts detailed error information including
    file name, line number, exception type, and traceback.
    """
    _, _, exc_tb = error_detail.exc_info()

    file_name = exc_tb.tb_frame.f_code.co_filename
    line_number = exc_tb.tb_lineno
    error_type = type(error).__name__
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    traceback_details = "".join(traceback.format_exception(*error_detail.exc_info()))

    error_message = (
        f"\n{'='*80}\n"
        f"Timestamp     : {timestamp}\n"
        f"Exception Type: {error_type}\n"
        f"File Name     : {file_name}\n"
        f"Line Number   : {line_number}\n"
        f"Error Message : {str(error)}\n"
        f"Traceback     :\n{traceback_details}"
        f"{'='*80}\n"
    )

    return error_message


class CustomException(Exception):
    """
    Custom exception class for detailed error reporting.
    """

    def __init__(self, error: Exception, error_detail: sys):
        self.error_message = get_error_details(error, error_detail)
        super().__init__(self.error_message)

    def __str__(self):
        return self.error_message