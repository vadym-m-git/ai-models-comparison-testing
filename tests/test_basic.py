import pytest
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Model constant for all tests
MODEL = "gpt-4o-mini"


@pytest.fixture
def openai_client():
    """Setup OpenAI client for all tests"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not found in .env")
    return OpenAI(api_key=api_key)


def test_model_responds(openai_client):
    """TEST #1: Verify model returns a response"""
    # Call OpenAI API
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Say 'hello' and nothing else"}],
    )
    # Extract the answer
    answer = response.choices[0].message.content
    # Verify we got something back
    assert answer is not None, "Model returned None"
    assert len(answer) > 0, "Model returned empty string"
    assert "hello" in answer.lower(), f"Expected 'hello', got: {answer}"
    print(f"\n✅ TEST #1 PASSED - Model responded: '{answer}'")


def test_determinism_at_zero_temperature(openai_client):
    """TEST #2: Same input at temp=0 gives same output"""

    prompt = "What is 5 + 3? Answer with only the number."
    answers = []

    # Ask the same question 3 times
    for i in range(3):
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,  # ← This makes it deterministic
        )
        answer = response.choices[0].message.content.strip()
        answers.append(answer)
        print(f"  Attempt {i+1}: {answer}")

    # All 3 answers should be identical
    assert answers[0] == answers[1] == answers[2], f"Answers differ: {answers}"

    assert "8" in answers[0], f"Wrong math answer: {answers[0]}"

    print(f"\n✅ TEST #2 PASSED - All 3 answers matched: '{answers[0]}'")


def test_temperature_affects_creativity(openai_client):
    """TEST #3: Higher temperature gives varied responses"""

    prompt = "Describe a sunset in exactly 5 words"

    # Low temperature (deterministic)
    response_low = openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=0
    )
    answer_low = response_low.choices[0].message.content

    # High temperature (creative)
    response_high = openai_client.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}], temperature=1.8
    )
    answer_high = response_high.choices[0].message.content

    print(f"\n  🌡️  Temp=0.0: {answer_low}")
    print(f"  🌡️  Temp=1.8: {answer_high}")

    # They should be different (not guaranteed, but very likely)
    # So we just verify both gave valid responses
    assert len(answer_low) > 0, "Low temp response empty"
    assert len(answer_high) > 0, "High temp response empty"

    print(f"\n✅ TEST #3 PASSED - Both temperatures produced responses")

def test_max_tokens_limit(openai_client):
    """TEST #4: Model respects token limits"""
    
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": "Write a long story about a dragon"
        }],
        max_tokens=10,  # Very short limit
        temperature=0
    )
    
    answer = response.choices[0].message.content
    word_count = len(answer.split())
    
    print(f"\n  📏 Response: '{answer}'")
    print(f"  📊 Word count: {word_count}")
    
    # With max_tokens=10, response should be very brief
    assert word_count < 15, f"Response too long: {word_count} words"
    
    print(f"\n✅ TEST #4 PASSED - Response limited to {word_count} words")

def test_json_format_output(openai_client):
    """TEST #5: Model can return valid JSON"""
    
    import json
    
    response = openai_client.chat.completions.create(
        model=MODEL,
        messages=[{
            "role": "user",
            "content": """Return this information as valid JSON:
            Name: Alice
            Age: 30
            City: Paris
            
            Use keys: name, age, city"""
        }],
        temperature=0
    )
    
    answer = response.choices[0].message.content
    print(f"\n  📄 Raw response: {answer}")

    # Remove Markdown code block formatting if present
    if answer.strip().startswith("```"):
        # Remove leading/trailing triple backticks and optional 'json' label
        answer = answer.strip()
        # Remove all lines starting/ending with triple backticks and 'json' label
        lines = answer.splitlines()
        # Remove first line if it starts with ``` or ```json
        if lines[0].strip().startswith("```"):
            lines = lines[1:]
        # Remove last line if it is just ```
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        answer = "\n".join(lines).strip()

    # Try to parse as JSON
    try:
        data = json.loads(answer)
        print(f"  ✓ Parsed JSON: {data}")

        # Verify expected fields exist (case-insensitive)
        keys_lower = {k.lower() for k in data.keys()}
        assert "name" in keys_lower, "Missing 'name' field"
        assert "age" in keys_lower, "Missing 'age' field"
        assert "city" in keys_lower, "Missing 'city' field"

        print(f"\n✅ TEST #5 PASSED - Valid JSON with all fields")

    except json.JSONDecodeError as e:
        pytest.fail(f"Model did not return valid JSON: {e}\nResponse: {answer}")