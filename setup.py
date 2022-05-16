# -*- coding: utf-8 -*-
import sys

from setuptools import setup, find_packages

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('service/version.py').read())

if sys.version_info < (3,):
    sys.exit('Sorry, Python3 is required.')

setup(
    name='text2vec-service',
    version=__version__,
    description='Mapping a variable-length sentence to a fixed-length vector using BERT model (Server)',
    url='https://github.com/shibing624/nlp-service',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    author='XuMing',
    author_email='xuming624@qq.com',
    license='Apache 2.0',
    packages=find_packages(),
    zip_safe=False,
    install_requires=[
        'six',
        'pyzmq>=17.1.0',
        'loguru',
        'GPUtil>=1.3.0',
        'text2vec',
        'transformers>=4.6.0',
        'numpy',
    ],
    extras_require={
        'pytorch': ['torch'],
        'http': ['flask', 'flask-compress', 'flask-cors', 'flask-json']
    },
    classifiers=(
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        "License :: OSI Approved :: Apache Software License",
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ),
    entry_points={
        'console_scripts': ['service-server-start=service.server.cli:main',
                            'service-server-benchmark=service.server.cli:benchmark',
                            'service-server-terminate=service.server.cli:terminate'],
    },
    keywords='bert, nlp, pytorch, machine learning, sentence encoding embedding serving',
)
