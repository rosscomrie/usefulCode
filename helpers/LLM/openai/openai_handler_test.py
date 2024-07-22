"""Unit tests for the OpenAIHandler class."""

import json
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from openai import APIError, AuthenticationError, RateLimitError

from openai_handler import (
    OpenAIHandler,
    OpenAIParameters,
)


@pytest.fixture()  # type: ignore[misc] # unit test being untyped due to decorator
def mock_openai_client() -> Generator[Mock, None, None]:
    """
    Fixture to mock the OpenAI client.

    Yields
    ------
    Mock
        A mocked OpenAI client.

    """
    with patch(
        "src.utils.helpers.agentic_framework.llm_handlers.openai_handler.OpenAI"
    ) as mock_openai:
        yield mock_openai


@pytest.fixture()  # type: ignore[misc] # unit test being untyped due to decorator
def mock_tiktoken() -> Generator[Mock, None, None]:
    """
    Fixture to mock the tiktoken library.

    Yields
    ------
    Mock
        A mocked tiktoken library.

    """
    with patch(
        "src.utils.helpers.agentic_framework.llm_handlers.openai_handler.tiktoken"
    ) as mock_tiktoken:
        mock_encoding = Mock()
        mock_encoding.encode.return_value = [1, 2, 3]  # Simulate 3 tokens for any input
        mock_tiktoken.encoding_for_model.return_value = mock_encoding
        yield mock_tiktoken


@pytest.fixture()  # type: ignore[misc] # unit test being untyped due to decorator
def handler(mock_openai_client: Mock, mock_tiktoken: Mock) -> OpenAIHandler:  # noqa: ARG001 (these arguments are used... remove them and check if the unit tests actually run :)
    """
    Fixture to create an instance of OpenAIHandler.

    Parameters
    ----------
    mock_openai_client : Mock
        The mocked OpenAI client.
    mock_tiktoken : Mock
        The mocked tiktoken library.

    Returns
    -------
    OpenAIHandler
        An instance of OpenAIHandler for testing.

    """
    return OpenAIHandler(
        api_key="test_key",
        model="gpt-3.5-turbo",
        enable_interaction_log=True,
        log_folder="logs/llm/test/pytest_logs/",
        conversation_folder="logs/llm/test/pytest_conversations",
    )


def test_initialization(handler: OpenAIHandler) -> None:
    """
    Test the initialization of OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    assert handler.api_key == "test_key"
    assert handler.model == "gpt-3.5-turbo"
    assert isinstance(handler.parameters, OpenAIParameters)
    assert handler.enable_interaction_log is True
    assert handler.log_folder == "logs/llm/test/pytest_logs/"
    assert handler.conversation_folder == "logs/llm/test/pytest_conversations"
    assert isinstance(handler.conversation_id, str)
    assert handler.message_id_counter == 0
    assert handler.use_streaming is False
    assert handler.always_confirm is True


def test_add_message(handler: OpenAIHandler) -> None:
    """
    Test adding a message to the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    handler.add_message("user", "Hello, AI!")
    expected_count = 3  # 3 tokens in "Hello, AI!"
    assert len(handler.messages) == 1
    assert handler.messages[0].role == "user"
    assert handler.messages[0].content == "Hello, AI!"
    assert handler.token_counts["user"] == expected_count
    assert handler.cumulative_input_tokens == expected_count


def test_add_system_message(handler: OpenAIHandler) -> None:
    """
    Test adding a system message to the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    handler.add_system_message("You are a helpful assistant.")
    expected_count = 3  # 3 tokens in "You are a helpful assistant."
    assert len(handler.messages) == 1
    assert handler.messages[0].role == "system"
    assert handler.messages[0].content == "You are a helpful assistant."
    assert handler.token_counts["system"] == expected_count
    assert handler.cumulative_input_tokens == expected_count


def test_clear_all_messages(handler: OpenAIHandler) -> None:
    """
    Test clearing all messages from the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    handler.add_user_message("Hello")
    handler.add_assistant_message("Hi there!")
    initial_input_tokens = handler.cumulative_input_tokens
    initial_output_tokens = handler.cumulative_output_tokens
    handler.clear_all_messages()
    assert len(handler.messages) == 0
    assert handler.message_id_counter == 0
    assert handler.token_counts == {"system": 0, "user": 0, "assistant": 0}
    assert handler.cumulative_input_tokens == initial_input_tokens
    assert handler.cumulative_output_tokens == initial_output_tokens


def test_delete_message(handler: OpenAIHandler) -> None:
    """
    Test deleting a message from the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    handler.add_user_message("First message")
    handler.add_assistant_message("Response")
    handler.add_user_message("Second message")
    expected_length = 2
    expected_count = 3
    expected_cumulative_tokens = 6

    with patch("builtins.input", return_value="y"):
        handler.delete_message("last", 1)

    assert len(handler.messages) == expected_length
    assert handler.messages[-1].content == "Response"
    assert (
        handler.token_counts["user"] == expected_count
    )  # Only "First message" remains
    assert handler.cumulative_input_tokens == expected_cumulative_tokens  # Unchanged


def test_chat_with_json_mode_disabled(
    handler: OpenAIHandler, mock_openai_client: Mock
) -> None:
    """
    Test the chat functionality of the OpenAIHandler with JSON mode disabled.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    str
        The response from the chat.

    """
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Hello, I'm an AI assistant."
    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_completion
    )

    handler.add_user_message("Hello, AI!")
    response = handler.chat()
    expected_length = 2

    assert isinstance(response, str)
    assert response == "Hello, I'm an AI assistant."
    assert len(handler.messages) == expected_length
    assert handler.messages[-1].role == "assistant"
    assert handler.messages[-1].content == "Hello, I'm an AI assistant."


def test_change_parameter_value(handler: OpenAIHandler) -> None:
    """
    Test changing a parameter value in the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    new_temperature = 0.8
    handler.change_parameter_value("temperature", new_temperature)
    assert handler.parameters.temperature == new_temperature


def test_change_model(handler: OpenAIHandler, mock_tiktoken: Mock) -> None:
    """
    Test changing the model in the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_tiktoken : Mock
        The mocked tiktoken library.

    """
    handler.change_model("gpt-4")
    assert handler.model == "gpt-4"
    mock_tiktoken.encoding_for_model.assert_called_with("gpt-4")


def test_count_tokens(handler: OpenAIHandler) -> None:
    """
    Test counting tokens in the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    Returns
    -------
    int
        The number of tokens counted.

    """
    assert handler.count_tokens("Test message")


def test_get_token_count(handler: OpenAIHandler) -> None:
    """
    Test getting token counts from the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    Returns
    -------
    Dict[str, int]
        A dictionary containing token counts for each role.

    """
    handler.add_user_message("User message")
    handler.add_assistant_message("Assistant response")
    assert handler.get_token_count()


def test_get_cumulative_tokens(handler: OpenAIHandler) -> None:
    """
    Test getting cumulative token counts from the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    Returns
    -------
    Dict[str, int]
        A dictionary containing cumulative token counts.

    """
    handler.add_user_message("User message")
    handler.add_assistant_message("Assistant response")
    cumulative_tokens = handler.get_cumulative_tokens()
    assert cumulative_tokens is not None


def test_chat(handler: OpenAIHandler, mock_openai_client: Mock) -> None:
    """
    Test the chat functionality of the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    str
        The response from the chat.

    """
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Hello, I'm an AI assistant."
    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_completion
    )

    handler.add_user_message("Hello, AI!")
    response = handler.chat()
    expected_length = 2

    assert response == "Hello, I'm an AI assistant."
    assert len(handler.messages) == expected_length
    assert handler.messages[-1].role == "assistant"
    assert handler.messages[-1].content == "Hello, I'm an AI assistant."
    assert handler.cumulative_input_tokens > 0
    assert handler.cumulative_output_tokens > 0


def test_chat_streaming(handler: OpenAIHandler, mock_openai_client: Mock) -> None:
    """
    Test the streaming chat functionality of the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    str
        The response from the streaming chat.

    """
    handler.use_streaming = True
    mock_chunk = Mock()
    mock_chunk.choices = [Mock()]
    mock_chunk.choices[0].delta.content = "Hello, "
    mock_openai_client.return_value.chat.completions.create.return_value = [
        mock_chunk,
        mock_chunk,
    ]

    handler.add_user_message("Hello, AI!")
    response = handler.chat()
    expected_length = 2

    assert response == "Hello, Hello, "
    assert len(handler.messages) == expected_length
    assert handler.messages[-1].role == "assistant"
    assert handler.messages[-1].content == "Hello, Hello, "


def test_save_and_load_conversation(handler: OpenAIHandler, tmp_path: Path) -> None:
    """
    Test saving and loading a conversation in the OpenAIHandler.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    tmp_path : Path
        A temporary path for saving and loading the conversation.

    """
    handler.conversation_folder = str(tmp_path)
    handler.add_user_message("Hello, AI!")
    handler.add_assistant_message("Hello, human!")

    handler.save_conversation()

    new_handler = OpenAIHandler(
        api_key="test_key", model="gpt-3.5-turbo", conversation_folder=str(tmp_path)
    )
    new_handler.load_conversation(handler.conversation_id)

    assert new_handler.messages == handler.messages
    assert new_handler.token_counts == handler.token_counts
    assert new_handler.cumulative_input_tokens == handler.cumulative_input_tokens
    assert new_handler.cumulative_output_tokens == handler.cumulative_output_tokens


def test_api_key_validation_success(mock_openai_client: Mock) -> None:
    """
    Test successful API key validation.

    Parameters
    ----------
    mock_openai_client : Mock
        The mocked OpenAI client.

    """
    mock_openai_client.return_value.models.list.return_value = None
    handler = OpenAIHandler(api_key="valid_key", model="gpt-3.5-turbo")
    assert handler.api_key == "valid_key"


def test_api_key_validation_failure(mock_openai_client: Mock) -> None:
    """
    Test API key validation failure.

    Parameters
    ----------
    mock_openai_client : Mock
        The mocked OpenAI client.

    """
    mock_response = Mock()
    mock_response.status_code = 401
    mock_openai_client.return_value.models.list.side_effect = AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )
    handler = OpenAIHandler(api_key="invalid_key", model="gpt-3.5-turbo")
    assert handler.api_key == "invalid_key"
    # Check if the client was not initialized due to invalid key
    assert not hasattr(handler, "client") or handler.client is None


def test_change_api_key_success(
    handler: OpenAIHandler, mock_openai_client: Mock
) -> None:
    """
    Test successful API key change.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    bool
        True if the API key change was successful, False otherwise.

    """
    mock_openai_client.return_value.models.list.return_value = None
    result = handler.change_api_key("new_valid_key")
    assert result is True
    assert handler.api_key == "new_valid_key"


def test_change_api_key_failure(
    handler: OpenAIHandler, mock_openai_client: Mock
) -> None:
    """
    Test API key change failure.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    bool
        True if the API key change was successful, False otherwise.

    """
    mock_response = Mock()
    mock_response.status_code = 401
    mock_openai_client.return_value.models.list.side_effect = AuthenticationError(
        message="Invalid API key",
        response=mock_response,
        body={"error": {"message": "Invalid API key"}},
    )
    result = handler.change_api_key("new_invalid_key")
    assert result is False
    assert handler.api_key == "test_key"  # Original key should be retained


def test_rate_limit_error(handler: OpenAIHandler, mock_openai_client: Mock) -> None:
    """
    Test handling of rate limit errors.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    Optional[str]
        The chat response, or None if a rate limit error occurred.

    """
    mock_response = Mock()
    mock_response.status_code = 429
    mock_openai_client.return_value.chat.completions.create.side_effect = (
        RateLimitError(
            message="Rate limit exceeded",
            response=mock_response,
            body={"error": {"message": "Rate limit exceeded"}},
        )
    )
    handler.add_user_message("Hello, AI!")
    response = handler.chat()
    assert response is None


def test_api_error(handler: OpenAIHandler, mock_openai_client: Mock) -> None:
    """
    Test handling of API errors.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    Optional[str]
        The chat response, or None if an API error occurred.

    """
    mock_openai_client.return_value.chat.completions.create.side_effect = APIError(
        message="API Error",
        request=Mock(),
        body={"error": {"message": "API Error", "type": "api_error"}},
    )
    handler.add_user_message("Hello, AI!")
    response = handler.chat()
    assert response is None


@pytest.fixture()  # type: ignore[misc] # unit test being untyped due to decorator
def json_handler(mock_openai_client: Mock, mock_tiktoken: Mock) -> OpenAIHandler:  # noqa: ARG001 (these arguments are used... remove them and check if the unit tests actually run :)
    """Fixture to create an instance of OpenAIHandler with JSON mode enabled."""
    return OpenAIHandler(
        api_key="test_key",
        model="gpt-3.5-turbo",
        enable_interaction_log=True,
        log_folder="logs/llm/test/pytest_logs/",
        conversation_folder="logs/llm/test/pytest_conversations",
        use_json_mode=True,
    )


def test_json_mode_initialization(json_handler: OpenAIHandler) -> None:
    """
    Test the initialization of OpenAIHandler with JSON mode enabled.

    Parameters
    ----------
    json_handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    assert json_handler.use_json_mode is True


def test_toggle_json_mode(handler: OpenAIHandler) -> None:
    """
    Test toggling JSON mode on and off.

    Parameters
    ----------
    handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    assert handler.use_json_mode is False
    handler.toggle_json_mode(True)
    assert handler.use_json_mode is True
    handler.toggle_json_mode(False)
    assert handler.use_json_mode is False


def test_chat_json_mode(json_handler: OpenAIHandler, mock_openai_client: Mock) -> None:
    """
    Test the chat functionality of the OpenAIHandler in JSON mode.

    Parameters
    ----------
    json_handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    Returns
    -------
    dict
        The JSON response from the chat.

    """
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = json.dumps(
        {"response": "Hello, I'm an AI assistant."}
    )
    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_completion
    )

    json_handler.add_user_message("Hello, AI!")
    response = json_handler.chat()
    expected_length = 2

    assert isinstance(response, dict)
    assert response == {"response": "Hello, I'm an AI assistant."}
    assert len(json_handler.messages) == expected_length
    assert json_handler.messages[-1].role == "assistant"
    assert (
        json_handler.messages[-1].content
        == '{"response": "Hello, I\'m an AI assistant."}'
    )
    assert json_handler.cumulative_input_tokens > 0
    assert json_handler.cumulative_output_tokens > 0


def test_chat_json_mode_invalid_response(
    json_handler: OpenAIHandler, mock_openai_client: Mock
) -> None:
    """
    Test functionality of the OpenAIHandler in JSON mode with an invalid JSON response.

    Parameters
    ----------
    json_handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    """
    mock_completion = Mock()
    mock_completion.choices = [Mock()]
    mock_completion.choices[0].message.content = "Invalid JSON response"
    mock_openai_client.return_value.chat.completions.create.return_value = (
        mock_completion
    )

    json_handler.add_user_message("Hello, AI!")
    response = json_handler.chat()

    assert response is None


def test_chat_json_mode_streaming(
    json_handler: OpenAIHandler, mock_openai_client: Mock
) -> None:
    """
    Test the streaming chat functionality of the OpenAIHandler in JSON mode.

    Parameters
    ----------
    json_handler : OpenAIHandler
        The OpenAIHandler instance to test.
    mock_openai_client : Mock
        The mocked OpenAI client.

    """
    json_handler.use_streaming = True
    mock_chunk1 = Mock()
    mock_chunk1.choices = [Mock()]
    mock_chunk1.choices[0].delta.content = '{"partial": "Hello'
    mock_chunk2 = Mock()
    mock_chunk2.choices = [Mock()]
    mock_chunk2.choices[0].delta.content = ", I'm an AI\"}"
    mock_openai_client.return_value.chat.completions.create.return_value = [
        mock_chunk1,
        mock_chunk2,
    ]

    json_handler.add_user_message("Hello, AI!")
    response = json_handler.chat()
    expected_length = 2

    assert response == {"partial": "Hello, I'm an AI"}
    assert len(json_handler.messages) == expected_length
    assert json_handler.messages[-1].role == "assistant"
    assert json_handler.messages[-1].content == '{"partial": "Hello, I\'m an AI"}'


def test_get_current_config_with_json_mode(json_handler: OpenAIHandler) -> None:
    """
    Test getting the current configuration with JSON mode enabled.

    Parameters
    ----------
    json_handler : OpenAIHandler
        The OpenAIHandler instance to test.

    """
    config = json_handler.get_current_config()
    assert config["model"] == "gpt-3.5-turbo"
    assert isinstance(config["parameters"], dict)
    assert config["json_mode"] is True
