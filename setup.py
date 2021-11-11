import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="brats",
    version="0.0.1",
    author="Bruno M. Pacheco",
    author_email="mpacheco.bruno@gmail.com",
    description="Useful functions and pipelines for brain tumor segmentation.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gama-ufsc/brats",
    project_urls={
        "Bug Tracker": "https://github.com/gama-ufsc/brats/issues",
    },
    package_dir={"": "."},
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",  # No idea what the actual requirements are
)
