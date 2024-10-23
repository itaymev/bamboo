# This file is used to package the code into a distributable format.
from setuptools import setup, find_packages

setup(
    name='Bamboo',
    version='0.1.0',
    author='Itay Mevorach',
    author_email='itaym@uoregon.edu',
    description='Advanced data cleaning built on top of Pandas',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_data_cleaner',  # Repo link
    packages=find_packages(),
    install_requires=[
        'pandas>=1.1.0',
        'scikit-learn>=0.24.0',
        'numpy>=1.18.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6',
)
