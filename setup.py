from setuptools import setup, find_packages


setup(
    name='PSA',
    version='1',
    license='MIT',
    author="Pablo Sartori",
    author_email='psartor@igc.gulbenkian.pt',
    packages=find_packages(),
    url='https://github.com/sartorilab/PSA',
    keywords=["biomechanics", "protein-structure", "biophysics", "bioinformatics"],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "biopython",
        "urllib3",
        "numba",
    ],
    py_modules=["psa/elastic", "psa/forms", "psa/load", "psa/sequence", "psa/spatial"],
)
