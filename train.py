#!/usr/bin/env python 
#一般要让程序执行时去寻找环境设置里对应的python安装路径，写程序的时候一般要这么写
"""
    Main Function.
"""

import torch
import os
import argparse # 动态输入必备
import random
from train_extractive import train_ext, validate_ext, test_ext
from others.logging import logger, init_logger


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-task", default = "ext", type = str, choices = ['ext', 'abs'])
    parser.add_argument("-encoder",default="bert",type=str,choices=["bert", " baseline"])
    parser.add_argument("-mode", default = 'train', type = str, choices = ['train', 'validate', 'test'])
    parser.add_argument("-bert_path",default="./bert_models/bert-base-uncased",type=str)
    parser.add_argument("-log_file",default = "./logs/dataset_train.log")
    parser.add_argument("-model_path", default = './models')
    parser.add_argument("-bert_data_path", default = "./bert_data")
    parser.add_argument("-results_path", default = "./results"  )
    parser.add_argument("-batch_size", default = 140, type = int)   
    parser.add_argument("-finetune_bert", type=str2bool, nargs='?', const=True, default=True) 
    parser.add_argument("-param_init", default=0, type=float)
    parser.add_argument("-param_init_glorot", type=str2bool, nargs='?',const=True,default=True)


    parser.add_argument("-train_from_ckpt", default = '')

    parser.add_argument("-gpu_ranks",default='0',type=str)
    parser.add_argument("-visible_gpus",default="-1",type=str, help="if -1 ,use cpu, else use gpu") #用在CUDA设备的指定上,即os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus 
    parser.add_argument('-seed', default = 1231, type = int)
    args = parser.parse_args() # 解析输入进来的参数，最终是一个类加属性的形式,对应的种类是add_argument()里的type
    init_logger(args.log_file)# 需要注意的是,一般是放在train或者多进程里，即需要长时间反馈日志的时候，使用，以及在总程序里使用
    args.gpu_ranks = [int(i) for i in args.gpu_ranks.split(",")]
    args.world_size = len(args.gpu_ranks)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus 
    device = "cpu" if args.visible_gpus == "-1" else "cuda"
    device_id = 0 if device == "cuda" else -1
    if args.task == 'abs':
        print(1)
    elif args.task == 'ext':
        if (args.mode == 'train'):
            train_ext(args,device,device_id)
