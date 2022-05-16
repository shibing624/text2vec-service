# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
import time
from loguru import logger
import sys

sys.path.append('..')
from service.client import BertClient

logger.add('test.log')


def test_get_emb():
    # encode a list of strings
    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    with BertClient(show_server_config=True) as bc:
        r = bc.encode(data)
        print(type(r), r, r[0], r.shape)
        logger.info(r.shape)
