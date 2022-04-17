import numpy as np
import cv2
import time
import logging
import os

def convert(image, screen_height, screen_width):
    image = cv2.resize(image, (screen_height, screen_width))
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)
    
    return image

def get_tensorboard_name():
    now = time.localtime()
    return str(now.tm_mon) +'_' + str(now.tm_mday)+ '_' + str(now.tm_hour) + '_'+ str(now.tm_min)


def get_logger(cfg):

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    logger_path = './logs/'+timestamp +'.log'

    check_makedirs(os.path.dirname(logger_path))

    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
    file_handler = logging.FileHandler(logger_path)
    logger.addHandler(file_handler)
    
    return logger

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)
def check_makedirs(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)