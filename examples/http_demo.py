# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""

import sys
import requests

if __name__ == '__main__':
    ip = sys.argv[1]
    http_port = int(sys.argv[2])
    r = requests.post(
        f"http://{ip}:{http_port}/encode",
        json={
            "id": 1,
            "texts": ["你好", "啥意思"],
        }
    )
    print(r.text)
