import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent
README = (HERE / "README.md").read_text()

setup(
    name="rrc_2022_datasets",
    version="0.1.0",
    description="Gym environments which provide offline RL datasets collected on the TriFinger system.",
    author="Nico Gürtler",
    author_email="nico.guertler@tuebingen.mpg.de",
    long_description=README,
    long_description_content_type="text/markdown",
    keywords=[
        "offline reinforcement learning", "TriFinger", "Real Robot Challenge", "dexterous manipulation"
    ],
    packages=find_packages(),
    install_requires=[
        "numpy", "gym", "h5py", "tqdm", "numpy-quaternion", "trifinger_simulation"
    ]
)
