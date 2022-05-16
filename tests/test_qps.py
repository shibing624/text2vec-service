# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import os
import time
from loguru import logger
from text2vec import SentenceModel
import sys

sys.path.append('..')

logger.add('test.log')
pwd_path = os.path.abspath(os.path.dirname(__file__))


def test_local_model_speed():
    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    num_tokens = sum([len(i) for i in data])
    model = SentenceModel('shibing624/text2vec-base-chinese')
    for j in range(10):
        tmp = data * (2 ** j)
        c_num_tokens = num_tokens * (2 ** j)
        start_t = time.time()
        r = model.encode(tmp)
        assert r is not None
        print('result shape', r.shape)
        time_t = time.time() - start_t
        logger.info('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
                    (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))
