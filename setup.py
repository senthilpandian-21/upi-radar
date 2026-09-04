"""Packaging metadata for UPI Radar (Razorpay AI Buildathon 2026 — Track 5)."""
from pathlib import Path

from setuptools import find_packages, setup

setup(
    name="upi-radar",
    version="1.0.0",
    description=(
        "AI-powered UPI outage prediction and smart transaction routing "
        "for Razorpay — predict payment outages BEFORE they happen."
    ),
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    author="UPI Radar Team",
    url="https://github.com/yourusername/upi-radar",
    python_requires=">=3.10",
    packages=find_packages(exclude=("tests", "notebooks", "dashboard")),
    install_requires=[
        # Core data
        "numpy>=1.24",
        "pandas>=1.5",
        "scikit-learn>=1.3",
        # API
        "fastapi>=0.100",
        "uvicorn>=0.23",
        "pydantic>=2.0",
        # Dashboard
        "streamlit>=1.30",
        "plotly>=5.15",
        # Infra clients
        "kafka-python>=2.0",
        "redis>=4.5",
        "sqlalchemy>=2.0",
        "apscheduler>=3.10",
        "requests>=2.31",
        "beautifulsoup4>=4.12",
        "reportlab>=4.0",
        "razorpay>=1.3",
        "python-dotenv>=1.0",
        "loguru>=0.7",
        "httpx>=0.24",
    ],
    extras_require={
        "ml": ["tensorflow>=2.13", "prophet>=1.1"],
        "agents": ["langgraph>=0.1", "celery>=5.3"],
        "alerts": ["twilio>=8.0"],
        "dev": ["pytest>=7.0", "pytest-asyncio>=0.21"],
    },
    entry_points={
        "console_scripts": [
            "upi-radar-api=api.main:run",
            "upi-radar-agent=agent.monitor_agent:main",
            "upi-radar-synthetic=data.synthetic.synthetic_generator:main",
        ],
    },
    include_package_data=True,
)
