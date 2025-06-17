import logging
import os

class Logger:
    def __init__(self, name='trainer', log_file='logs/train.log', level=logging.INFO):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        if not self.logger.handlers:
            self.logger.addHandler(file_handler)

    def get_logger(self):
        return self.logger