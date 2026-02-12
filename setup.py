# setup.py
from setuptools import setup, find_packages

setup(
    name="qwen-rag-assistant",
    version="1.0.0",
    description="Qwen-RAG自动化助手 - 支持文档问答、鼠标自动化和对话记忆",
    author="Your Name",
    packages=find_packages(),
    install_requires=[
        line.strip() for line in open("requirements.txt").readlines()
    ],
    entry_points={
        "console_scripts": [
            "qwen-rag=main:main",
        ]
    },
    python_requires=">=3.9",
)