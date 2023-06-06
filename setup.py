from setuptools import setup, find_packages


setup(
    name='proFSA',
    version='0.13',
    license='MIT',
    author="Pablo Sartori",
    author_email='psartor@igc.gulbenkian.pt',
    packages=find_packages(),
    url='https://github.com/sartorilab/proFSA',
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
    py_modules=["fsa/elastic", "fsa/forms", "fsa/load", "fsa/sequence", "fsa/spatial"],
)
