from setuptools import find_packages, setup


NAME = "spkanon_eval"
DESCRIPTION = "Speaker anonymization Evaluation (SpAnE)"
URL = "https://github.com/carlosfranzreb/spkanon_eval"
EMAIL = "carlos.franzreb@dfki.de"
AUTHOR = "Carlos Franzreb"
REQUIRES_PYTHON = ">=3.9.0"
VERSION = "0.3.0"


def req_file(filename: str) -> list[str]:
    """Get requirements from file"""
    with open(filename, encoding="utf-8") as f:
        content = f.readlines()
    required, links = list(), list()
    for line in content:
        line = line.strip()
        required.append(line)
    return required, links


REQUIRED = req_file("requirements.txt")
EXTRAS = {}
VERSION = "0.1.0"


with open("README.md", encoding="utf-8") as f:
    long_description = "\n" + f.read()


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    include_package_data=True,
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
