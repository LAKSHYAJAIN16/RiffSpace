"""
Setup script for RiffSpace.

Install: pip install -e .
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read requirements
requirements = []
requirements_file = Path(__file__).parent / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip() 
            for line in f 
            if line.strip() and not line.startswith('#')
        ]

# Read README
readme = Path(__file__).parent / "README.md"
long_description = readme.read_text() if readme.exists() else ""

setup(
    name="riffspace",
    version="0.3.0",
    description="Musical Structure Retrieval via Vector Embeddings",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="RiffSpace Research",
    author_email="",
    url="",
    packages=find_packages(),
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=7.4.0',
            'black>=23.0.0',
            'flake8>=6.0.0',
        ],
        'vectordb': [
            'faiss-cpu>=1.7.4',
            'pinecone-client>=2.2.0',
            'chromadb>=0.4.0',
        ],
        'ui': [
            'streamlit>=1.28.0',
            'plotly>=5.14.0',
        ],
        'audio': [
            'openl3>=0.4.0',
        ]
    },
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords='music, vector-database, embeddings, audio, similarity-search, information-retrieval',
    project_urls={
        'Documentation': '',
        'Source': '',
        'Bug Reports': '',
    },
)
