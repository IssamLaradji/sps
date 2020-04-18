from setuptools import setup

setup(name='sps',
      version='0.6.0',
      description='Stochastic Polyak Step-size',
      url='git@github.com:IssamLaradji/sps.git',
      maintainer='Issam Laradji',
      maintainer_email='issam.laradji@gmail.com',
      license='MIT',
      packages=['sps'],
      zip_safe=False,
      install_requires=[
        'tqdm>=0.0'
        'numpy>=0.0',
        'pandas>=0.0',
        'Pillow>=0.0',
        'scikit-image>=0.0',
        'scikit-learn>=0.0',
        'scipy>=0.0',
        'sklearn>=0.0',
        'torch>=0.0',
        'torchvision>=0.0',
      ]),