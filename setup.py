from setuptools import setup, find_packages


setup(
        name='pyxBA',
        version='0.1.0',
        description='Tools for caclulating expected batting averages.',
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
