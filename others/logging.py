#!/usr/bin/env python
"""
    Log the process of training, validating or testing.
"""
# from __future__ import absolute_import
# 绝对导入避免子包覆盖掉标准库模块
# 可以认为相对导入经常用.  .. ...来指代目录，绝对导入要指定相应包

import logging
logger = logging.getLogger()

def init_logger(log_file = None, log_file_level = logging.NOTSET):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    log_format = logging.Formatter("[%(asctime)s %(leveltime)s] %(message)s")
    # streamhandler 一般输出到终端，需要flush，而filehandler需要输出到磁盘中
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(log_format)
    logger.addHandler(sh)##这里也可以直接用logger.handlers = [handler]
    
    if log_file and log_file != '':
        fh = logging.FileHandler(log_file)# 对应指定的路径
        fh.setLevel(log_file_level)
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    return logger





if __name__ == "__main__":
    
