from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


setup(
    name="gpumonitor",
    version="0.1.0",
    description="GPU Monitoring Callbacks for TensorFlow and PyTorch Lightning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Raphael Meudec",
    author_email="raphael.meudec@gmail.com",
    url="https://github.com/sicara/gpumonitor",
    license="MIT",
    install_requires=["gpustat>=0.6.0"],
    extras_require={
        "publish": ["bumpversion>=0.5.3", "twine>=1.13.0"],
    },
    packages=find_packages(),
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Operating System :: OS Independent",
    ],
)
