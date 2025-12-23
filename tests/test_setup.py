import os
from dotenv import load_dotenv

def test_environment_setup():
    """Verify environment is configured correctly"""
    load_dotenv()
    
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    assert openai_key is not None, "OPENAI_API_KEY not found in .env"
    assert anthropic_key is not None, "ANTHROPIC_API_KEY not found in .env"
    assert openai_key.startswith("sk-"), "Invalid OpenAI key format"
    assert anthropic_key.startswith("sk-ant-"), "Invalid Anthropic key format"
    
    print("\n✅ Environment setup correct!")
    print(f"✅ OpenAI key: {openai_key[:10]}...")
    print(f"✅ Anthropic key: {anthropic_key[:15]}...")

def test_imports():
    """Verify all required packages are installed"""
    import pytest
    import openai
    import anthropic
    import sklearn
    
    print("\n✅ All packages imported successfully!")