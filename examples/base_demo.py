#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XuMing(xuming624@qq.com)

import sys
import time

sys.path.append('..')
from service.client import BertClient

if __name__ == '__main__':
    # encode a list of strings
    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    with BertClient(show_server_config=True) as bc:
        r = bc.encode(data)
        print(type(r), r, r[0], r.shape)
