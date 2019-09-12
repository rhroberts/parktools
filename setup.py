from setuptools import setup, find_packages


setup(
        name='parktools',
        version='0.1.0',
        description='Python tools for the ballpark',
        author="Rusty Roberts",
        license="GNU GPL v3",
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
