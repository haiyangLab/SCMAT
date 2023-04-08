#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2022/4/16 20:01
# @Author  : Chiancc

import os
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter


class Logger(object):
    def __init__(self, log_dir, log_hist=True):
        """
        Create a summary writer logging to log_dir
        Args:
            log_dir:  target directory
            log_hist:  record log history for every time
        """
        if log_hist:
            log_dir = os.path.join(
                log_dir,
                datetime.datetime.now().strftime("%Y_%m_%d__%H_%M_%S"))
        self.writer = SummaryWriter(log_dir)

    def one_scalar(self, tag, scalar_value, global_step):
        """
        Log a scalar variable.
        Args:
            tag (string): Data identifier
            scalar_value (float or string): Value to save
            global_step (int): Global step value to record
        """
        self.writer.add_scalar(tag, scalar_value, global_step)

    def scalars(self, tag_value_pairs, global_step):
        """
        Log scalar variables
        Args:
            tag_value_pairs:
            global_step:
        """
        for tag, value in tag_value_pairs:
            self.writer.add_scalar(tag, value, global_step)


class MyLogging:
    # 初始化日志
    def __init__(self, log_path, log_level='debug'):
        self.logger = logging.getLogger("my_logger")

        self.level = logging.DEBUG
        if log_level == 'info':
            self.level = logging.INFO

        self.logger.setLevel(self.level)

        logging.basicConfig(level=logging.DEBUG,
                            format='[%(asctime)s] [%(filename)s]-->[%(levelname)s] %(message)s',
                            datefmt='%Y-%m-%d %H:%M:%S',
                            filename=log_path,
                            filemode='a')

        console_format = logging.Formatter(
            '[%(asctime)s] [%(filename)s]-->[%(levelname)s] %(message)s')  # [line:%(lineno)d]

        # filehandler = logging.FileHandler(log_path)
        # filehandler.setLevel(self.level)
        # filehandler.setFormatter(file_format)

        streamhandler = logging.StreamHandler()
        streamhandler.setLevel(logging.INFO)
        streamhandler.setFormatter(console_format)

        # self.logger.addHandler(filehandler)
        self.logger.addHandler(streamhandler)

        self.logger.info("------logger inited-------")

    def get_logger(self):
        return self.logger


log_path = "./run.log"
log = MyLogging(log_path).get_logger()
