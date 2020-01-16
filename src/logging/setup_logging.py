import os.path
import logging
import logging.config
import sys


def setup_logging(level = logging.INFO, file_name = None, file_mode = "w"):

    log_format = '%(asctime)s - %(message)s'
    datefmt = '%d-%b-%y %H:%M:%S'

    if file_name is None:
        logging.basicConfig(
            format=log_format,
            datefmt=datefmt,
            level=level
        )
    else:
        logging.basicConfig(
            filename=file_name,
            filemode=file_mode,
            format=log_format,
            datefmt=datefmt,
            level=level
        )

        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    if os.path.isfile("logger.conf"):
        logging.debug("Found 'logger.conf'")
        try:
            logging.config.fileConfig(fname="logger.conf", disable_existing_loggers=False)
        except Exception as e:
            logging.warning(f"Couldn't load 'logger.conf': {e}")
