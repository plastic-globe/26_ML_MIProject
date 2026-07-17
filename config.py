"""Project-wide configuration values."""

import os

# Set HF_TOKEN in the environment before running Hugging Face based experiments.
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# Optional API settings used by utils/APIKeyTester.py.
OPENAI_KEYS = []
OHMYGPT_KEY = ""
ZHIZENGZENG_KEY = ""
OHMYGPT_URLS = []
ZHIZENGZENG_URL = ""
OPENAI_URL = "https://api.openai.com/v1"
