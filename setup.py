from setuptools import setup, find_packages

setup(
    name='airplane_sat',
    version='0.1.0',
    packages=find_packages(include=['airplane_sat', 'airplane_sat.*']),
    install_requires=[
        'sklearn>=0.0',
    ],
)