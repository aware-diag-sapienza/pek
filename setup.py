from distutils.core import setup
from pathlib import Path

from pek.version import __version__

if __name__ == "__main__":
    setup(
        name="pec",
        version=__version__,
        description="PEK - Progressive Ensemble K-Means Clustering",
        url="https://aware-diag-sapienza.github.io/pek",
        author="A.WA.RE Research Group - Sapienza Università di Roma",
        author_email="graziano.blasilli@uniroma1.it",
        license="GNU General Public License (GPL)",
        platforms=["Linux", "Mac OS-X", "Solaris", "Unix", "Windows"],
        packages=["pek", "pek.csv"],
        # python_requires='>=3.9.0',
        install_requires=open(Path("requirements.txt")).read().strip().split("\n"),
        package_data={"pek.csv": ["*.csv"]},
        include_package_data=True,
        zip_safe=True,
        # download_url=f"https://github.com/aware-diag-sapienza/pec/archive/refs/tags/v{__version__}.tar.gz",
        # long_description=open(Path("PyPI.md")).read(),
        # long_description_content_type="text/markdown",
    )
