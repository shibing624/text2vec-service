# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import torch
import sys

sys.path.append('../..')
from service.server import BertServer
from service.server.benchmark import run_benchmark
from service.server.helper import (
    get_run_args, get_benchmark_parser,
    get_shutdown_parser
)


def main():
    torch.multiprocessing.set_start_method('spawn')
    with BertServer(get_run_args()) as server:
        server.join()


def benchmark():
    torch.multiprocessing.set_start_method('spawn')
    args = get_run_args(get_benchmark_parser)
    run_benchmark(args)


def terminate():
    args = get_run_args(get_shutdown_parser)
    BertServer.shutdown(args)


if __name__ == '__main__':
    main()
