#!/usr/bin/env python
# -*- coding: utf-8 -*-
# XuMing(xuming624@qq.com)

# simple similarity search on FAQ
import sys
import numpy as np

sys.path.append('..')
from service.client import BertClient

topk = 5
questions = [
    '如何更换花呗绑定银行卡',
    '花呗更改绑定银行卡',
    '一个戴着安全帽的男人在跳舞',
    '一个小孩在骑马',
    '孩子在骑马',
    '一个女人在切洋葱',
    '一个人在切洋葱',
    '人们在玩板球',
    '男人们在打板球',
]
print('%d questions loaded, avg. len of %d' % (len(questions), np.mean([len(d) for d in questions])))

bc = BertClient()
doc_vecs = bc.encode(questions)

while True:
    query = input('your question: ')
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:topk]
    print('top %d questions similar to "%s"' % (topk, query))
    for idx in topk_idx:
        print('> %s\t%s' % ('%.1f' % score[idx], questions[idx]))
