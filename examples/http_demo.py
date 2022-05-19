# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import requests
import time


def get_encode_data(ip, http_port, sentences):
    r = requests.post(
        f"http://{ip}:{http_port}/encode",
        json={
            "id": 1,
            "texts": sentences,
        }
    )
    return r.json()


if __name__ == '__main__':
    ip = sys.argv[1]
    http_port = int(sys.argv[2])

    data = ['如何更换花呗绑定银行卡',
            '花呗更改绑定银行卡']
    print("data:", data)
    r = get_encode_data(ip, http_port, data)
    print(type(r.json()), r.json())

    num_tokens = sum([len(i) for i in data])
    for j in range(10):
        tmp = data * (2 ** j)
        c_num_tokens = num_tokens * (2 ** j)
        start_t = time.time()
        get_encode_data(ip, http_port, tmp)
        time_t = time.time() - start_t
        print('encoding %d sentences, spend %.2fs, %4d samples/s, %6d tokens/s' %
              (len(tmp), time_t, int(len(tmp) / time_t), int(c_num_tokens / time_t)))
