from setuptools import setup, find_packages


setup(
    name='proFSA',
    version='0.1',
    license='MIT',
    author="Pablo Sartori",
    author_email='psartor@igc.gulbenkian.pt',
    packages=find_packages(),
    #package_dir={'': 'src'},
    url='https://github.com/sartorilab/proFSA',
    keywords=["biomechanics", "protein-structure", "biophysics", "bioinformatics"],
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy",
        "scipy",
        "biopython",
        "urllib3",
    ],

)
