import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="til-final",
    version="2022.0.1",
    author="Je Hon Tan",
    author_email="jehontan@gmail.com",
    description="Final robotics challenge for TIL2022",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jehontan/til-final",
    project_urls={
        "Bug Tracker": "https://github.com/jehontan/til-final/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        'numpy >= 1.20',
        'scipy >= 1.7.0',
        'opencv-contrib-python >= 4.5',
        'matplotlib >= 3.1.2',
        'onnxruntime-gpu >= 1.10.0',
        'urllib3 >= 1.25.8',
        'pyyaml >= 5.3',
        'librosa >= 0.9.1',
        'tensorflow >= 2.8.0',
        'termcolor',
        'flask'
    ],
    entry_points = {
        'console_scripts': [
            'til-simulator=tilsim.simulator:main',
            'til-scoring=tilscoring.service:main',
            'til-judge=tilscoring.visualizer:main',
        ]
    },
)