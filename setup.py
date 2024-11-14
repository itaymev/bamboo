# setup.py
from setuptools import setup, find_packages

setup(
    name='BambooChute',
    version='1.0.1',
    author='Itay Mevorach',
    author_email='itaym@uoregon.edu',
    description='Advanced data cleaning built on top of Pandas',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/itaymev/smart-clean-package.git', 
    packages=find_packages(),
    install_requires=[
        'pandas>=1.1.0',
        'numpy>=1.18.0',
        'scikit-learn>=0.24.0', 
        'fancyimpute>=0.7.0'   
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.12',
    ],
    python_requires='>=3.6',
)
