from setuptools import setup
from setuptools import find_packages

setup(name='tpne-xgb',
      version='1.0',
      description='Temporally-pretained Node Embedder',
      author='Hafizh Adi Prasetya, Xin Liu, Akiyoshi Matono',
      author_email='hafizhadi.prasetya@aist.go.jp',
      download_url='https://github.com/hafizhadi/multiround-promo-fraud',
      license='MIT',
      install_requires=['dgl>=2.0.0',
                        'networkx>=3.1',
                        'numpy>=1.24.3',
                        'pandas>=2.2.1',
                        'scikit-learn>=1.3.0',
                        'scipy>=1.12.0',
                        'seaborn>=0.12.2',
                        'torch>=2.2.2',
                        'torch_geometric>=2.5.3',
                        'xgboost>=2.0.3',
                        ],
      package_data={'src': ['README.md', 'scripts/*.json']},
      packages=find_packages())