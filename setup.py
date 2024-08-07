# David Kaplan
from setuptools import setup
def readme():
    with open('README.md') as f:
        return f.read()

setup(
    name = "CSD",
    version = 1.0,
    description = "Python script to download code chef solutions for a problem",
    author = "Ganesh Kumar M",
    author_email = "ganeshkumarm.1996@gmail.com",
    license = "MIT",
    url = "https://github.com/GaneshmKumar/CSD",
    packages = ["csd"],
    install_requires=[
          'requests==2.17.3',
          'beautifulsoup4==4.6.0',
          'argparse==1.1'
      ],
    entry_points={
        'console_scripts':[
            'csd = csd.csd:main'
            ]
        },
    long_description=readme(),
)