#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XuMing(xuming624@qq.com)

# using BertClient in multicast way

import sys
import threading

sys.path.append('..')
from service.client import BertClient

def client_clone(id, idx):
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]), identity=id)
    for j in bc.fetch():
        print('clone-client-%d: received %d x %d' % (idx, j.shape[0], j.shape[1]))


if __name__ == '__main__':
    bc = BertClient(port=int(sys.argv[1]), port_out=int(sys.argv[2]))
    # start two cloned clients sharing the same identity as bc
    for j in range(2):
        t = threading.Thread(target=client_clone, args=(bc.identity, j))
        t.start()

    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    for _ in range(3):
        vec = bc.encode(data)
        print('bc received %d x %d' % (vec.shape[0], vec.shape[1]))
