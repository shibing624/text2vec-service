# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
from sentence_transformers import SentenceTransformer


def build_model(model_dir):
    try:
        logger.info('model dir: %s' % model_dir)
        model = SentenceTransformer(model_dir)
        return model
    except Exception as e:
        logger.error(f'fail to build model!, {e}', exc_info=True)
