"""Sets up the CoFi package for installation."""

#
# Copyright (c) Simon Kohaut, Felix Divo, and contributors
#
# This file is part of CoFi and licensed under the BSD 3-Clause License.
# You should have received a copy of the BSD 3-Clause License along with CoFi.
# If not, see https://opensource.org/license/bsd-3-clause/.
#

import re

import setuptools

# Find CoFi version and author strings
with open("cofi/__init__.py", encoding="utf8") as fd:
    content = fd.read()
    version = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)
    author = re.search(r'^__author__\s*=\s*[\'"]([^\'"]*)[\'"]', content, re.MULTILINE).group(1)

# Import readme
with open("README.md", encoding="utf8") as readme:
    long_description = readme.read()

setuptools.setup(
    name="python-cofi",
    version=version,
    author=author,
    author_email="simon.kohaut@cs.tu-darmstadt.de",
    description="A Python package for Bayesian estimation of agents that follow a constitution.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/simon-kohaut/CoFi",
    packages=setuptools.find_packages(exclude=["tests", "examples"]),
    package_data={
        "cofi": ["py.typed"],  # https://www.python.org/dev/peps/pep-0561/
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=[
        # particle filtering
        "numpy",
        "scipy",
        "promis==4.0.0.dev2",
    ],
    extras_require={
        # Running the examples in examples/
        "examples": [
            "pandas",
            "matplotlib",
            "pillow",
            "imageio-ffmpeg",
        ],
        # Development tools for quality assurance
        "dev": [
            # static code analysis
            "black",
            "ruff",
            # dynamic code analysis
            "pytest",
        ],
    },
    include_package_data=True,
)
