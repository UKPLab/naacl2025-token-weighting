import logging
import sys

# Global logger instance
logger = logging.getLogger("MyLogger")

def initialize_logger(log_file=None, rank=0):
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()  # Console handler
    # Set levels for handlers
    c_handler.setLevel(logging.DEBUG)
    # Create formatters and add them to the handlers
    c_format = logging.Formatter(f'%(name)s - %(levelname)s - {rank} - %(message)s')
    c_handler.setFormatter(c_format)

    # Clear existing handlers and add new ones
    logger.handlers.clear()
    logger.addHandler(c_handler)
    if log_file is not None:
        f_handler = logging.FileHandler(log_file)
        f_handler.setLevel(logging.DEBUG)  # File handler
        f_format = logging.Formatter(f'%(asctime)s - %(name)s - %(levelname)s - {rank} - %(message)s')
        f_handler.setFormatter(f_format)
        logger.addHandler(f_handler)
        logging.basicConfig(level=logging.DEBUG, handlers=[f_handler])

    logger.propagate = True
    logger.handlers[0].flush = sys.stdout.flush
