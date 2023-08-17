import os
import setuptools

setuptools.setup(
    name="trep",
    version="0.1.0",
    author="Stanley Risch",
    author_email="s.risch@fz-juelich.de",
    description="Module for region land eligibility analysis based on GLAES.",
    install_requires=[
        "numpy",
        "pandas",
        "glaes",
        "reskit",
        "geokit",
        "FINE",
        "MATES",
        "sqlalchemy"
        ]
    )
