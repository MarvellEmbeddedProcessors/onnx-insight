# SPDX-License-Identifier: BSD-3-Clause
# Copyright(C) 2023 Marvell.

from setuptools import setup

setup(name='onnxinsight',
      version='1.0.0',
      description='ONNX Insight, tool for ONNX model analysis',
      license='BSD-3-Clause',
      packages=['onnxinsight', 'onnxinsight.analy_func'],
      entry_points={
          'console_scripts': ['onnxinsight=onnxinsight.onnxinsight:main', ]
      },
      install_requires=[
          'onnx>=1.11.0',
          'numpy>=1.22.3',
          'onnxruntime>=1.10.0',
          'onnxsim>=0.3.10',
          'rich>=12.4.4',
      ],
      zip_safe=False
      )
