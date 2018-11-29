import logging
import os


# Absolute path to the file directory
FILE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)
LOG_FILE_NAME = 'data_writing.log'
LOG_DIRECTORY = os.path.abspath(
    os.path.join(FILE_DIRECTORY, *[os.pardir, 'logs', LOG_FILE_NAME])
)
logger_formatter = logging.Formatter('%(asctime)s : %(name)s : %(message)s')
file_handler = logging.FileHandler(LOG_DIRECTORY)
file_handler.setFormatter(logger_formatter)
logger.addHandler(file_handler)


def write_points_to_file(file_path, data):
    """
    Write (x,y) points to the specified file path
    :param file_path: Full path to the file
    :param data: Data in format [(x1, y1), ... ,(xn, yn)]
    :return:
    """
    logger.debug(f'Writing data to file {file_path}')

    with open(file_path,'w') as f:
        f.write("x,y\n")
        for d in data:
            f.write(f'{d[0]},{d[1]}\n')

    logger.debug('Data has been successfully written to file')
