from setuptools import setup
from setuptools import find_packages

setup(name='tpne-xgb',
      version='1.1',
      description='Multi-round adversarial graph-based promo fraud detection with an intelligent fraud generator',
      author='karneish',
      author_email='karneish.sk@gmail.com',
      download_url='https://github.com/karneish/enhanced-multiround-promo-fraud',
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