#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XuMing(xuming624@qq.com)

# using BertClient in sync way

import sys
import time

sys.path.append('..')
from service.client import BertClient

if __name__ == '__main__':
    # encode a list of strings
    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    num_tokens = sum([len(i) for i in data])
    with BertClient(show_server_config=True) as bc:
        r = bc.encode(data)
        print(r)
    bc = BertClient()
    for j in range(10):
        tmp = data * (2 ** j)
        c_num_tokens = num_tokens * (2 ** j)
        start_t = time.time()
        bc.encode(tmp)
        time_t = time.time() - start_t
        print('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
              (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))
