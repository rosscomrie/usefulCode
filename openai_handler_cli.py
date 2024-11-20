import os
import sys

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from helpers.logger.logger import Logger
from helpers.LLM.openai.openai_handler import OpenAIHandler

logger = Logger(initial_level="INFO", module_name="OpenAI Handler CLI")


def main():
    api_var_name = "OPEN_AI_API_KEY"
    api_key = os.getenv(f"{api_var_name}")
    if not api_key:
        logger.error(f"Please set the {api_var_name} environment variable.")
        return

    helper = OpenAIHandler(
        api_key=api_key,
        model="gpt-3.5-turbo",
        enable_interaction_log=True,
        use_streaming=True,
        log_folder="logs/test_logs",
        conversation_folder="logs/test_conversations",
        always_confirm=True,
    )

    if not hasattr(helper, "client"):
        logger.error(
            "Failed to initialize OpenAIHelper. Please check the logs for more information."
        )
        return

    logger.info("OpenAI Helper Test Script")
    logger.info("=========================")

    while True:
        print("\nOptions:")
        print("1. Chat with the model")
        print("2. Change model parameters")
        print("3. Save conversation")
        print("4. Load conversation")
        print("5. Delete messages")
        print("6. Clear all messages in current conversation")
        print("7. Start a new conversation")
        print("8. Print current messages")
        print("9. Display token count")
        print("10. Display cumulative token count")
        print("11. Change API key")
        print("E. Exit")

        choice = input("Enter your choice (1-11 or E): ")

        if choice == "1":
            user_input = input("Enter your message: ")
            helper.add_user_message(user_input)
            response = helper.chat()
            if response:
                logger.info(f"Assistant: {response}")
            else:
                logger.error(
                    "Failed to get a response. Please check the logs for more information."
                )

        elif choice == "2":
            param = input(
                "Enter parameter to change (top_p, temperature, max_tokens): "
            )
            value = float(input("Enter new value: "))
            helper.change_parameter_value(param, value)
            logger.info(f"Parameter {param} changed to {value}")

        elif choice == "3":
            helper.save_conversation()
            logger.info(f"Conversation saved with ID: {helper.conversation_id}")

        elif choice == "4":
            conv_id = input("Enter conversation ID to load: ")
            helper.load_conversation(conv_id)
            logger.info(f"Conversation {conv_id} loaded")

        elif choice == "5":
            number = int(input("Enter number of messages to delete: "))
            helper.delete_message("last", number)

        elif choice == "6":
            helper.clear_all_messages()
            logger.info("All messages cleared")

        elif choice == "7":
            helper.new_conversation()
            logger.info("New conversation started")

        elif choice == "8":
            helper.print_messages()

        elif choice == "9":
            token_count = helper.get_token_count()
            logger.info("Token Count by Role:")
            logger.info(f"System tokens: {token_count['system']}")
            logger.info(f"User tokens: {token_count['user']}")
            logger.info(f"Assistant tokens: {token_count['assistant']}")
            logger.info(f"Total tokens: {sum(token_count.values())}")

        elif choice == "10":
            cumulative_tokens = helper.get_cumulative_tokens()
            logger.info("Cumulative Token Count:")
            logger.info(
                f"Cumulative input tokens: {cumulative_tokens['cumulative_input_tokens']}"
            )
            logger.info(
                f"Cumulative output tokens: {cumulative_tokens['cumulative_output_tokens']}"
            )
            logger.info(
                f"Cumulative total tokens: {cumulative_tokens['cumulative_total_tokens']}"
            )

        elif choice == "11":
            new_api_key = input("Enter new API key: ")
            if helper.change_api_key(new_api_key):
                logger.info("API key successfully changed.")
            else:
                logger.error(
                    "Failed to change API key. Please check the logs for more information."
                )

        elif choice.upper() == "E":
            logger.info("Exiting...")
            break

        else:
            logger.info("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()