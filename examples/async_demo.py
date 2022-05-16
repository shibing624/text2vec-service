#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XuMing(xuming624@qq.com)
# using BertClient in async way

import sys
import time

sys.path.append('..')
from service.client import BertClient


def send_without_block(bc, data, repeat=10):
    # encoding without blocking:
    print('sending all data without blocking...')
    for _ in range(repeat):
        bc.encode(data, blocking=False)
    print('all sent!')


if __name__ == '__main__':
    bc = BertClient()
    num_repeat = 20

    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    send_without_block(bc, data, num_repeat)
    num_expect_vecs = len(data) * num_repeat

    # then fetch all
    print('now waiting until all results are available...')
    vecs = bc.fetch_all(concat=True)
    print('received %s, expected: %d' % (vecs.shape, num_expect_vecs))

    # now send it again
    send_without_block(bc, data, num_repeat)

    # this time fetch them one by one, due to the async encoding and server scheduling
    # sending order is NOT preserved!
    for v in bc.fetch():
        print(v)
        print('received %s, shape %s' % (v.id, v.embedding.shape))

    # finally let's do encode-fetch at the same time but in async mode
    # we do that by building an endless data stream, generating data in an extremely fast speed
    def text_gen():
        while True:
            yield data

    for j in bc.encode_async(text_gen(), max_num_batch=20):
        print('received %d : %s' % (j.id, j.embedding))
