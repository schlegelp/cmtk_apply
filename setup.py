from setuptools import setup, find_packages

setup(
    name="cmtk-apply",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.19.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    author="Philipp Schlegel",
    author_email="pms70@cam.ac.uk",
    description="Python library to read and apply CMTK transformations",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/philipp-schlegel/cmtk_apply",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
    ],
)
