import setuptools


with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
    name="teyered",
    version="0.1.1",
    author="Karolis Spukas, Kacper Kielak, Leonardo Castorina",
    author_email="hello@osmitau.com",
    description="Machine Learning and Computer Vision framework for tiredness detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/K-Kielak/teyered",
    packages=[
        "resources",
        "teyered",
        "teyered.data_processing",
        "teyered.data_processing.blinks",
        "teyered.data_processing.pose",
        "teyered.io"
    ],
    install_requires=[
        "cmake", "dlib", "imutils", "numpy", "opencv-python"
    ],
    python_requires=">=3.6",
    setup_requires=["pytest-runner"],
    tests_require=[
      "pytest",
      "pytest-mock",
      "PyHamcrest"
    ],
    package_data={
        "resources": ["*"]
    },
    classifiers=[
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
    ],
)