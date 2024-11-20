from helpers.logger.logger import Logger
from helpers.LLM.openai.openai_handler import OpenAIHandler
import os

logger = Logger(initial_level="DEBUG", module_name="OpenAI Handler CLI")

api_key = os.getenv("OPEN_AI_API_KEY")

helper = OpenAIHandler(
    api_key=api_key,
    model="gpt-3.5-turbo",
    enable_interaction_log=True,
    use_streaming=True,
    log_folder="logs/test_logs",
    conversation_folder="logs/test_conversations",
    always_confirm=True,
)

helper.add_system_message("You are an expert joke writer")
helper.add_user_message("Tell me a funny joke")
helper.print_messages()
logger.debug("This is a gap")
helper.chat()
helper.print_messages()