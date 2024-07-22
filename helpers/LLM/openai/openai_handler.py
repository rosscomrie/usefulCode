"""Class that handles and manages an OpenAI Agent."""

from __future__ import annotations

import csv
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path

# Type checking imports
from typing import TYPE_CHECKING

import tiktoken
from openai import APIError, AuthenticationError, OpenAI, RateLimitError

if TYPE_CHECKING:
    from collections.abc import Iterator

    from openai.types.chat import ChatCompletion, ChatCompletionChunk

logger = Logger(initial_level="INFO") # Need to import logger lines for custom logger class


@dataclass
class OpenAIParameters:
    """
    Parameters for OpenAI API requests.

    Attributes
    ----------
    top_p : float
        Top-p sampling parameter.
    temperature : float
        Temperature for controlling randomness in responses.
    max_tokens : int
        Maximum number of tokens to generate.

    """

    top_p: float = 1.0
    temperature: float = 0.7
    max_tokens: int = 150


@dataclass
class Message:
    """
    Represents a message in the conversation.

    Attributes
    ----------
    id : int
        Unique identifier for the message.
    role : str
        Role of the message sender (e.g., "user", "assistant", "system").
    content : str
        Content of the message.
    timestamp : datetime
        Timestamp of when the message was created.

    """

    id: int
    role: str
    content: str
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, int | str]:
        """
        Convert the message to a dictionary.

        Returns
        -------
        Dict[str, Union[int, str]]
            Dictionary representation of the message.

        """
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, str | int]) -> Message:
        """
        Create a Message instance from a dictionary.

        Parameters
        ----------
        data : Dict[str, Union[str, int]]
            Dictionary containing message data.

        Returns
        -------
        Message
            New Message instance.

        """
        timestamp_data = data["timestamp"]
        if isinstance(timestamp_data, int):
            timestamp = datetime.fromtimestamp(timestamp_data, UTC)
        else:
            timestamp = datetime.fromisoformat(str(timestamp_data))

        return cls(
            id=int(data["id"]),
            role=str(data["role"]),
            content=str(data["content"]),
            timestamp=timestamp,
        )


class OpenAIHandler:
    """
    Helper class for interacting with OpenAI API.

    Attributes
    ----------
    api_key : str
        OpenAI API key.
    model : str
        Name of the OpenAI model to use.
    messages : List[Message]
        List of messages in the conversation.
    parameters : OpenAIParameters
        Parameters for OpenAI API requests.
    total_tokens : int
        Total number of tokens used in the conversation.
    enable_interaction_log : bool
        Whether to enable logging of interactions.
    log_folder : str
        Folder to store log files.
    conversation_folder : str
        Folder to store conversation files.
    conversation_id : str
        Unique identifier for the current conversation.
    message_id_counter : int
        Counter for generating unique message IDs.
    use_streaming : bool
        Whether to use streaming for API requests.
    always_confirm : bool
        Whether to always confirm before deleting messages.
    client : OpenAI
        OpenAI client instance.
    tokenizer : tiktoken.Encoding
        Tokenizer for the current model.
    log_file : str
        Path to the log file for the current conversation.

    """

    def __init__(  # noqa: PLR0913 (too many arguments apparently... this is a central class so I don't care.)
        self,
        api_key: str,
        model: str,
        messages: list[Message] | None = None,
        parameters: OpenAIParameters | None = None,
        enable_interaction_log: bool = False,
        log_folder: str = "logs/llm/handlers",
        conversation_folder: str = "logs/llm/conversations",
        use_streaming: bool = False,
        always_confirm: bool = True,
        use_json_mode: bool = False,
    ) -> None:
        """
        Initialize the OpenAIHelper.

        Parameters
        ----------
        api_key : str
            OpenAI API key.
        model : str
            Name of the OpenAI model to use.
        messages : Optional[List[Message]], optional
            Initial list of messages, by default None.
        parameters : Optional[OpenAIParameters], optional
            Parameters for OpenAI API requests, by default None.
        enable_interaction_log : bool, optional
            Whether to enable logging of interactions, by default False.
        log_folder : str, optional
            Folder to store log files, by default "logs/llm/handlers".
        conversation_folder : str, optional
            Folder to store conversation files, by default "logs/llm/conversations".
        use_streaming : bool, optional
            Whether to use streaming for API requests, by default False.
        always_confirm : bool, optional
            Whether to always confirm before deleting messages, by default True.
        use_json_mode : bool, optional
            Whether to enable JSON responses for API responses, by default False.

        """
        self.api_key = api_key
        self.model = model
        self.messages: list[Message] = messages or []
        self.parameters = parameters or OpenAIParameters()
        self.total_tokens: int = 0
        self.enable_interaction_log = enable_interaction_log
        self.log_folder = log_folder
        self.conversation_folder = conversation_folder
        self.conversation_id = str(uuid.uuid4())
        self.message_id_counter: int = 0
        self.use_streaming = use_streaming
        self.always_confirm = always_confirm
        self.token_counts: dict[str, int] = {"system": 0, "user": 0, "assistant": 0}
        self.cumulative_input_tokens: int = 0
        self.cumulative_output_tokens: int = 0
        self.use_json_mode = use_json_mode

        Path(self.log_folder).mkdir(exist_ok=True, parents=True)
        Path(self.conversation_folder).mkdir(exist_ok=True, parents=True)

        self.log_file = os.path.join(self.log_folder, f"{self.conversation_id}.txt")

        if self._validate_api_key():
            self.client = OpenAI(api_key=self.api_key)
            self.tokenizer = tiktoken.encoding_for_model(model)
            self._log_event(
                "initialization",
                {
                    "conversation_id": self.conversation_id,
                    "model": model,
                    "parameters": json.dumps(asdict(self.parameters)),
                },
            )
        else:
            logger.error("Failed to initialize OpenAIHelper due to invalid API key.")

    def _validate_api_key(self) -> bool:
        """
        Validate the API key.

        Returns
        -------
        bool
            True if the API key is valid, False otherwise.

        """
        try:
            test_client = OpenAI(api_key=self.api_key)
            test_client.models.list()
            return True  # noqa: TRY300 (explicit return in private method)
        except AuthenticationError:
            logger.exception(
                "Invalid API key. Please check your OpenAI API key and try again."
            )
        except APIError:
            logger.exception("API Error occurred")
        except Exception:
            logger.exception("An unexpected error occurred while validating API key:")
        return False

    def _log_event(
        self, event_type: str, details: dict[str, str | float | bool]
    ) -> None:
        """
        Log an event to the log file.

        Parameters
        ----------
        event_type : str
            Type of the event.
        details : Dict[str, str | float | bool]
            Details of the event.

        """
        timestamp = datetime.now(timezone.utc).isoformat()  # noqa: UP017 (timezone issues)
        details_str = json.dumps(details)

        with Path(self.log_file).open("a", newline="") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow([timestamp, event_type, details_str])

    def chat(self) -> str | dict | None:
        """
        Send a chat request to the OpenAI API and process the response.

        Returns
        -------
        Optional[Union[str, dict]]
            The generated response from the assistant, or None if an error occurred.
            Returns a dict if JSON mode is enabled, otherwise a string.

        """
        try:
            input_messages = [
                {"role": msg.role, "content": msg.content} for msg in self.messages
            ]

            input_tokens = sum(
                self.count_tokens(msg["content"]) for msg in input_messages
            )
            self.cumulative_input_tokens += input_tokens

            chat_params = {
                "model": self.model,
                "messages": input_messages,
                "stream": self.use_streaming,
                **asdict(self.parameters),
            }

            if self.use_json_mode:
                chat_params["response_format"] = {"type": "json_object"}

            if TYPE_CHECKING:
                response: ChatCompletion | Iterator[ChatCompletionChunk] = (
                    self.client.chat.completions.create(**chat_params)
                )
            else:
                response = self.client.chat.completions.create(**chat_params)

            if self.use_streaming:
                output = "".join(
                    chunk.choices[0].delta.content or "" for chunk in response
                )
            else:
                output = response.choices[0].message.content  # type: ignore[union-attr] # Iterator does have choices attribute as defined by OpenAI :)

            output_tokens = self.count_tokens(output)
            self.cumulative_output_tokens += output_tokens

            self.add_assistant_message(output)

            if self.use_json_mode:
                try:
                    return json.loads(output)
                except json.JSONDecodeError:
                    logger.exception(f"Failed to parse JSON response: {output}")
                    return None
            else:
                return output  # (explicit in try/except block)

        except (AuthenticationError, RateLimitError, APIError):
            logger.exception("API error occurred:")
        except Exception:
            logger.exception("An unexpected error occurred:")

        return None

    ### Message Management ###

    def _generate_message_id(self) -> int:
        """
        Generate a unique message ID.

        Returns
        -------
        int
            Unique message ID.

        """
        self.message_id_counter += 1
        return self.message_id_counter

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation and update token counts.

        Parameters
        ----------
        role : str
            Role of the message sender.
        content : str
            Content of the message.

        """
        message_id = self._generate_message_id()
        message = Message(id=message_id, role=role, content=content)
        self.messages.append(message)
        self._log_event(
            "message_added", {"id": message_id, "role": role, "content": content}
        )

        tokens = self.count_tokens(content)
        self.token_counts[role] += tokens
        if role in ["system", "user"]:
            self.cumulative_input_tokens += tokens
        elif role == "assistant":
            self.cumulative_output_tokens += tokens

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation.

        Parameters
        ----------
        content : str
            Content of the system message.

        """
        self.add_message("system", content)

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation.

        Parameters
        ----------
        content : str
            Content of the user message.

        """
        self.add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """
        Add an assistant message to the conversation.

        Parameters
        ----------
        content : str
            Content of the assistant message.

        """
        self.add_message("assistant", content)

    def clear_all_messages(self) -> None:
        """Clear all messages in conversation and reset non-cumulative token counts."""
        self.messages.clear()
        self.message_id_counter = 0
        self.token_counts = {"system": 0, "user": 0, "assistant": 0}
        self._log_event("all_messages_cleared", {})

    def delete_message(self, strategy: str, number: int) -> None:
        """
        Delete messages based on the specified strategy.

        Parameters
        ----------
        strategy : str
            The strategy to use for deletion. Currently supports 'last'.
        number : int
            The number of messages to delete.

        Raises
        ------
        ValueError
            If an unsupported delete strategy is provided
            or if attempting to delete a system message.

        """
        if self.always_confirm:
            confirmation = input(
                f"Are you sure you want to delete {number} message(s) "
                f"using the '{strategy}' strategy? (y/n): "
            )
            if confirmation.lower() != "y":
                logger.info("Deletion cancelled.")
                return

        deleted_ids = []
        if strategy.lower() == "last":
            for _ in range(number):
                if self.messages and self.messages[-1].role != "system":
                    deleted_message = self.messages.pop()
                    deleted_ids.append(deleted_message.id)
                    # Update token counts (but don't decrease cumulative counts)
                    self.token_counts[deleted_message.role] -= self.count_tokens(
                        deleted_message.content
                    )
                else:
                    err = (
                        "Cannot delete system message. Clear all messages "
                        "or start a new conversation instead."
                    )
                    logger.error(err)
        else:
            err = f"Unsupported delete strategy: {strategy}"
            logger.error(err)

        self._log_event(
            "messages_deleted",
            {
                "strategy": strategy,
                "number": number,
                "deleted_ids": json.dumps(deleted_ids),
            },
        )
        logger.info(f"Deleted {number} message(s) using the '{strategy}' strategy.")

    def print_messages(self) -> None:
        """Print all messages in the conversation."""
        for message in self.messages:
            logger.info(f"{message.role.title()}: {message.content}")

    ### Conversation Management ###

    def new_conversation(self) -> None:
        """Start a new conversation and reset all token counts."""
        self.messages.clear()
        self.token_counts = {"system": 0, "user": 0, "assistant": 0}
        self.cumulative_input_tokens = 0
        self.cumulative_output_tokens = 0
        self.message_id_counter = 0
        self.conversation_id = str(uuid.uuid4())
        self._log_event("new_conversation", {})
        self.log_file = os.path.join(self.log_folder, f"{self.conversation_id}.txt")

    def save_conversation(self) -> None:
        """Save the current conversation to a JSON file."""
        conversation_data = {
            "conversation_id": self.conversation_id,
            "model": self.model,
            "parameters": asdict(self.parameters),
            "messages": [msg.to_dict() for msg in self.messages],
            "token_counts": self.token_counts,
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
        }
        filename = os.path.join(
            self.conversation_folder, f"{self.conversation_id}.json"
        )
        with Path(filename).open("w") as f:
            json.dump(conversation_data, f, indent=2)
        self._log_event("conversation_saved", {"filename": filename})

    def load_conversation(self, conversation_id: str) -> None:
        """
        Load a conversation from a JSON file.

        Parameters
        ----------
        conversation_id : str
            ID of the conversation to load.

        """
        filename = os.path.join(self.conversation_folder, f"{conversation_id}.json")
        with Path(filename).open() as f:
            conversation_data = json.load(f)

        self.conversation_id = conversation_data["conversation_id"]
        self.model = conversation_data["model"]
        self.parameters = OpenAIParameters(**conversation_data["parameters"])
        self.messages = [
            Message.from_dict(msg) for msg in conversation_data["messages"]
        ]
        self.message_id_counter = (
            max(msg.id for msg in self.messages) if self.messages else 0
        )
        self.tokenizer = tiktoken.encoding_for_model(self.model)
        self.token_counts = conversation_data.get(
            "token_counts", {"system": 0, "user": 0, "assistant": 0}
        )
        self.cumulative_input_tokens = conversation_data.get(
            "cumulative_input_tokens", 0
        )
        self.cumulative_output_tokens = conversation_data.get(
            "cumulative_output_tokens", 0
        )

        self._log_event(
            "conversation_loaded",
            {"filename": filename, "conversation_id": self.conversation_id},
        )

    ### Class Settings Management ###

    def get_current_config(self) -> dict[str, str | dict[str, float | int] | bool]:
        """
        Get the current configuration.

        Returns
        -------
        Dict[str, Union[str, Dict[str, Union[float, int]]]]
            Dictionary containing the current model and parameters.

        """
        return {
            "model": self.model,
            "parameters": asdict(self.parameters),
            "json_mode": self.use_json_mode,
            "streaming": self.use_streaming,
        }

    def change_parameter_value(self, parameter: str, value: float) -> None:
        """
        Change the value of a parameter.

        Parameters
        ----------
        parameter : str
            Name of the parameter to change.
        value : Union[float, int]
            New value for the parameter.

        Raises
        ------
        ValueError
            If an invalid parameter is provided.

        """
        if hasattr(self.parameters, parameter):
            old_value = getattr(self.parameters, parameter)
            setattr(self.parameters, parameter, value)
            self._log_event(
                "parameter_changed",
                {"parameter": parameter, "old_value": old_value, "new_value": value},
            )
        else:
            err = f"Invalid parameter: {parameter}"
            logger.error(err)

    def change_model(self, new_model: str) -> None:
        """
        Change the OpenAI model.

        Parameters
        ----------
        new_model : str
            Name of the new model to use.

        """
        old_model = self.model
        self.model = new_model
        self.tokenizer = tiktoken.encoding_for_model(new_model)
        self._log_event(
            "model_changed", {"old_model": old_model, "new_model": new_model}
        )

    def change_api_key(self, new_api_key: str) -> bool:
        """
        Change the API key and validate it.

        Parameters
        ----------
        new_api_key : str
            The new API key to use.

        Returns
        -------
        bool
            True if the API key was successfully changed and validated, False otherwise.

        """
        old_api_key = self.api_key
        self.api_key = new_api_key
        if self._validate_api_key():
            self.client = OpenAI(api_key=self.api_key)
            self._log_event(
                "api_key_changed",
                {"old_key": old_api_key[-4:], "new_key": new_api_key[-4:]},
            )
            logger.info("API key successfully changed and validated.")
            changed = True
        else:
            self.api_key = old_api_key
            logger.error("Failed to change API key. Reverting to the previous key.")
            changed = False
        return changed

    def toggle_json_mode(self, enable: bool) -> None:
        """
        Toggle JSON mode on or off.

        Parameters
        ----------
        enable : bool
            True to enable JSON mode, False to disable it.

        """
        self.use_json_mode = enable
        self._log_event("json_mode_toggled", {"enabled": enable})
        logger.info(f"JSON mode {'enabled' if enable else 'disabled'}.")

    ### Token Management ###

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the given text.

        Parameters
        ----------
        text : str
            Text to count tokens for.

        Returns
        -------
        int
            Number of tokens in the text.

        """
        return len(self.tokenizer.encode(text))

    def get_token_count(self) -> dict[str, int]:
        """
        Get the token count for the conversation by role.

        Returns
        -------
        Dict[str, int]
            Dictionary containing token counts for system, user, and assistant roles.

        """
        return self.token_counts

    def get_cumulative_tokens(self) -> dict[str, int]:
        """
        Get the cumulative token counts for input and output.

        Returns
        -------
        Dict[str, int]
            Dictionary containing cumulative input, output, and total token counts.

        """
        return {
            "cumulative_input_tokens": self.cumulative_input_tokens,
            "cumulative_output_tokens": self.cumulative_output_tokens,
            "cumulative_total_tokens": self.cumulative_input_tokens
            + self.cumulative_output_tokens,
        }
