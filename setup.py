from setuptools import setup, find_packages

setup(
    name="tictactoe-ai-analysis",
    version="0.1.0",
    description="Tic-Tac-Toe AI Analysis with Reinforcement Learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
    ],
    python_requires=">=3.8",
)