from setuptools import setup, find_packages

setup(
    name="ANCOVA",  # Nombre del paquete
    version="0.1.0",  # Versión inicial
    author="Germán Vallejo Palma",
    author_email="vallejopalma.g@gmail.com",
    description="Package for ANCOVA analysis and visualization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/GERMAN00VP/ANCOVA",  # Reemplaza con la URL de tu repositorio
    packages=find_packages(),  # Encuentra automáticamente los subpaquetes
    install_requires=[
        "numpy",
        "pandas",
        "statsmodels",
        "scipy",
        "matplotlib",
        "seaborn",
        "scikit_posthocs"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Requiere Python 3.6 o superior
)
