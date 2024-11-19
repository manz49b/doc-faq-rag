import unittest
from unittest.mock import patch, MagicMock
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src.claude import create_message  
from dotenv import load_dotenv
load_dotenv()  # Load environment variables from.env file

class TestCreateMessage(unittest.TestCase):

    @patch('src.claude.anthropic.Anthropic')
    def test_create_message_success(self, mock_anthropic):
        # Set up the mock response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(content="Hello, world!")

        # Call the function
        response = create_message("Tell me a joke.", 50)

        # Assert the response
        self.assertEqual(response, "Hello, world!")
        mock_anthropic.assert_called_once()
        mock_client.messages.create.assert_called_once_with(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[{"role": "user", "content": "Tell me a joke."}]
        )

    @patch('os.environ.get')
    def test_create_message_no_api_key(self, mock_get):
        # Mock the environment variable to simulate no API key
        mock_get.return_value = ''  # Simulate missing API key

        response = create_message("Tell me a joke.", 50)
        self.assertIsNone(response)

    @patch('src.claude.anthropic.Anthropic')
    def test_create_message_exception(self, mock_anthropic):
        # Test exception handling in the API call
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.side_effect = Exception("API error")

        response = create_message("Tell me a joke.", 50)
        self.assertIsNone(response)

    @patch('src.claude.anthropic.Anthropic')
    def test_create_message_empty_response(self, mock_anthropic):
        # Test when the API returns an empty response
        mock_client = MagicMock()
        mock_anthropic.return_value = mock_client
        mock_client.messages.create.return_value = MagicMock(content="")

        response = create_message("Tell me a joke.", 50)
        self.assertIsNone(response)

if __name__ == "__main__":
    unittest.main()
