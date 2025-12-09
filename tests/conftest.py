import pytest
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

@pytest.fixture
def openai_client():
    """Setup OpenAI client for all tests"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)
