#!/usr/bin/env python
"""
# Author: Xia Chenrui
# Created Time : Thu 18 Oct 2019 

# File Name: setup.py
# Description:

"""

from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(name='deeptcr',
      version='0.0.2',
      description='Single-Cell TCR seqences classify via deep learning',
      packages=find_packages(),

      author='Xia Chenrui',
      author_email='huhansan666666@gmail.com',
      url='https://github.com/huhansan666666/deepbcr',
      scripts=['TCR.py'],
      install_requires=requirements,
      python_requires='>3.6.0',

      classifiers=[
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.7',
          'Operating System :: MacOS :: MacOS X',
          'Operating System :: Microsoft :: Windows',
          'Operating System :: POSIX :: Linux',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
     ],
     )
