#!/usr/bin/env python
"""
# Author: Xia Cheurui
# Created Time :OCT 27 2019

# File Name: TCR.py
# Description: Single-Cell TCR seqences classify via deep learning.
    Input: 
        TCR seqences
    Output:
		Cell type
"""
import time
import torch 
import torchvision
import torchtext
from torch.utils.data import DataLoader

import numpy as np 
import pandas as pd 
import os
import argparse
import sys

from deeptcr import model
from deeptcr import utils
from deeptcr import test

if __name__ == '__main__':

    cur_path = os.path.abspath(os.curdir)

    parse=argparse.ArgumentParser(description='This is deeptcr')
    # (type,nargs,action,default,help)
    parse.add_argument("--learning_rate",'-l',type=float,default=0.01,help="Test1")
    parse.add_argument('--gpu', '-g', default=0, type=int, help='Select gpu device number when training')
    parse.add_argument('--picture',action='store_true',help='Draw the picture')
    parse.add_argument('--outdir', '-o', type=str, default=os.path.join(cur_path,'output'), help='Output path')
    parse.add_argument('--workdir', '-w', type=str, default=cur_path, help='Work path')
    parse.add_argument('--status', '-s', type=str, default='final_test', help='Choose your working status')
    parse.add_argument("--epoch", '-e', type=int,default=5, help="Tell thr program how many epochs you want")
    # parse.add_argument("--model", '-m', type=int,default=5, help="Tell thr program how many epochs you want")


    args = parse.parse_args()

    print(args)
    print(args.learning_rate)

    if torch.cuda.is_available(): # cuda device
        device='cuda'
        torch.cuda.set_device(args.gpu)
    else:
        device='cpu'

    outdir = args.outdir
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    if args.status == 'final_test':
    	test.final_test()
    else:
        output = model.run(args.epoch)
        print(output)
        print(type(output))


