from setuptools import setup, find_packages


setup(
        name='parktools',
        version='0.1.0',
        description='Tools for the ballpark',
        packages=find_packages(),
        install_requires=[
            'pandas',
            'scipy',
            'numpy',
            'matplotlib',
            'sklearn',
            'pybaseball'
    ]
)
