# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import requests

if __name__ == '__main__':
    ip = sys.argv[1]
    port = int(sys.argv[2])
    port_out = int(sys.argv[3])
    r = requests.post(
        f"http://{ip}:{port}/encode",
        json={
            "id": 1,
            "texts": ["你好", "啥意思"],
        }
    )
    print(r.text)
