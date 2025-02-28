from setuptools import setup, find_packages

setup(
    name='portfolio-optimizer',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A portfolio optimization tool based on Markowitz\'s Modern Portfolio Theory and Black-Litterman model.',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'yfinance',
        'pyyaml'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)