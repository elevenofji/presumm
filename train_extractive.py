#!/usr/bin/env python


import torch
import os
import glob
import random
import signal
import time


from models import data_loader, model_builder
from models.model_builder import ExtSummarizer
from others.logging import logger, init_logger

model_flags = ['hidden_size', 'ff_size', 'heads', 'inter_layers', 'encoder', 'ff_actv', 'use_interval', 'rnn_size']

def train_multi_ext(args):
    pass

def validate_ext(args, device_id):
    pass

def train_ext(args, device,device_id):
    if args.world_size > 1:
        train_multi_ext(args)
    else:
        train_single_ext(args, device, device_id)

def train_single_ext(args, device, device_id):
    init_logger(args.log_file)

    logger.info("Device ID {}".format(device_id))
    logger.info("Device {}".format(device))


    if device_id >= 0:
        torch.cuda.set_device(device_id)
        torch.cuda.manual_seed(args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = True

    if args.train_from_ckpt != '':
        logger.info('Load ckpt from'.format(args.train_from_ckpt))
        ckpt = torch.load(args.train_from_ckpt, map_location = lambdastorage, loc : storage)

        ckpt_args = vars(ckpt['ckpt_args'])
        for k in ckpt_args.keys():
            if k in model_flags:
                setattr(args, k, ckpt_args[k])
    else:
        ckpt = None

    model = ExtSummarizer(args, device, ckpt)


