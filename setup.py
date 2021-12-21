import os
from setuptools import setup

with open(os.path.join(os.path.dirname(__file__), "README.md")) as fh:
    long_description = fh.read()

setup(
    name='thetafit',
    version='0.0.1',
    packages=['thetafit'],
    url='https://github.com/solbes/thetafit',
    download_url = 'https://github.com/solbes/thetafit/archive/refs/tags/0.0.1.tar.gz',
    license='MIT',
    author='Antti Solonen',
    author_email='antti.solonen@gmail.com',
    description='Parameter estimation for nonlinear models',
    keywords=['bayes', 'nonlinear', 'parameter estimation'],
    install_requires=['numpy', 'scipy', 'pandas', 'matplotlib'],
    extras_require={
        'dev': ['pytest']
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: MIT License",
        'Programming Language :: Python :: 3',
    ],
    long_description=long_description,
    long_description_content_type="text/markdown",
)
