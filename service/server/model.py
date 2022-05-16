# -*- coding: utf-8 -*-
"""
@author:XuMing(xuming624@qq.com)
@description: 
"""
from loguru import logger
from text2vec import SentenceModel


def build_model(model_dir):
    try:
        logger.info('model_dir: %s' % model_dir)
        model = SentenceModel(model_dir)
        return model
    except Exception as e:
        logger.error(f'fail to build model!, {e}', exc_info=True)
