import pytest
from unittest.mock import patch, MagicMock
from services.generator import GeneratorService

def test_evaluate_groundedness():
    generator = GeneratorService()
    # High score > threshold (0.1)
    assert generator.evaluate_groundedness(0.85) is True
    # Low score < threshold
    assert generator.evaluate_groundedness(0.05) is False

def test_create_strict_system_prompt():
    generator = GeneratorService()
    context = "My name is John Doe."
    prompt = generator.create_strict_system_prompt(context)
    
    assert "You are a high-precision RAG assistant" in prompt
    assert "CONTEXT:\nMy name is John Doe." in prompt

@patch("services.generator.OpenAI")
def test_generate_stream_valid_client(mock_openai):
    # Setup mock
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    
    # Mock stream response
    mock_chunk1 = MagicMock()
    mock_chunk1.choices[0].delta.content = "Answer "
    mock_chunk2 = MagicMock()
    mock_chunk2.choices[0].delta.content = "to query."
    
    mock_client.chat.completions.create.return_value = [mock_chunk1, mock_chunk2]
    
    generator = GeneratorService()
    generator.client = mock_client
    
    context_chunks = [{"text": "Sample Context", "section_path": "Root"}]
    stream = generator.generate_stream("Test Query", context_chunks)
    
    result = list(stream)
    assert result == ["Answer ", "to query."]
    
def test_generate_stream_no_client():
    generator = GeneratorService()
    generator.client = None
    
    stream = generator.generate_stream("Test Query", [])
    result = list(stream)
    
    assert "NOT FOUND" in result[0]
