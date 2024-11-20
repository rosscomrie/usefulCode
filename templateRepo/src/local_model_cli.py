"""
Local Model CLI for Chat Completion and Fine-Tuning.

This module provides a command-line interface for interacting with and fine-tuning
local language models. It includes functionality for running chat completions,
fine-tuning models, and managing GPU resources.

Key Features:
- Interactive and non-interactive chat modes
- Model fine-tuning with custom training data
- GPU availability checking
- Flexible command-line argument parsing for model configuration

Usage:
    Run the script with desired command-line arguments to start a chat session
    or perform model fine-tuning.

Notes:
    Ensure all required dependencies are installed and the correct model files
    are available in the specified directories.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import TrainingArguments

# Add the parent directory of 'src' to the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from logger import Logger
from local_model import Model

logger = Logger(initial_level="DEBUG", module_name="Local Model CLI")


def check_gpu() -> torch.device:
    """
    Check for GPU availability and return the appropriate device.

    This function checks if a GPU is available for PyTorch computations.
    It prints information about the PyTorch version, CUDA version, and the selected device.

    Returns:
    -------
    torch.device
        The device to be used for computations (either 'cuda' or 'cpu').

    Notes:
    -----
    If you're stuck with a CPU, you might want to go make a coffee. Or ten.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("GPU is available. Using CUDA!")
    else:
        device = torch.device("cpu")
        logger.info("GPU not available. Using CPU. Prepare for a long wait, mortal.")

    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    return device


def run_chat(model: Model, generation_kwargs: dict, interactive: bool = True) -> None:
    """Run an interactive chat session with the given model."""
    # Set up the initial dialog with appropriate system message

    system_msg = "You are a helpful assistant."
        
    dialog = [{"role": "system", "content": system_msg}]
    
    logger.info("Starting chat. Type 'exit' to end the conversation.")
    while True:
        try:
            if interactive:
                user_input = input("You: ")
                logger.info(f"User: {user_input}")
                if user_input.lower() == "exit":
                    break
            else:
                user_input = input("Enter user message (or 'exit' to end): ")
                if user_input.lower() == "exit":
                    break

            dialog.append({"role": "user", "content": user_input})
            logger.info(f"Dialog: {dialog}")
            
            dialog, assistant_response = model.chat_completion(
                dialog, 
                **generation_kwargs
            )
            logger.info(f"Assistant: {assistant_response}", colour="ORANGE")

        except (ValueError, TypeError) as e:
            logger.exception(f"An exception occurred: {e}")
            logger.info("Continuing with the chat...")

        if not interactive:
            logger.info("\nCurrent dialog:")
            for message in dialog:
                logger.info(f"{message['role'].capitalize()}: {message['content']}\n")



def run_fine_tuning(
    model_name: str, fine_tuned_dir: str, training_data: list[dict[str, Any]]
) -> None:
    """
    Run fine-tuning on a given model with provided training data.

    This function initializes a ModelFineTuner, sets up training arguments,
    runs the fine-tuning process, and then cleans up resources.

    Parameters:
    ----------
    model_name : str
        The name of the model to fine-tune.
    fine_tuned_dir : str
        The directory where the fine-tuned model will be saved.
    training_data : list[dict[str, Any]]
        A list of dictionaries containing the training data.

    Returns:
    -------
    None

    Notes:
    -----
    - The function uses default training arguments which may need to be adjusted
      based on the specific requirements of the model and dataset.
    - After fine-tuning, the function calls `stop()` and `clear_memory()` on the
      fine_tuner to ensure proper resource cleanup.

    Raises:
    ------
    Any exceptions raised by ModelFineTuner or the fine-tuning process will
    propagate up from this function.
    """
    fine_tuner = ModelFineTuner(model_name, fine_tuned_dir)
    training_args = TrainingArguments(
        output_dir=fine_tuned_dir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        save_steps=5,
        save_total_limit=2,
    )
    fine_tuner.run_fine_tuning(training_data, training_args)
    fine_tuner.stop()
    fine_tuner.clear_memory()


def main() -> None:
    """
    Run chat completion with a local model.

    This function sets up command-line arguments, initializes the model,
    and runs the chat completion process. It handles GPU checking,
    model initialization, and proper cleanup of resources.

    Command-line Arguments:
    ----------------------
    --model_name: Name of the model to use (default: "meta-llama/Llama-3.1-8B-Instruct")
    --tokenizer_name: Name of the tokenizer to use (default: "meta-llama/Llama-3.1-8B-Instruct")
    --output_dir: Directory to load the model from (default: "src/models/base/Meta_Llama_3.1_8B_Instruct")
    --max_new_tokens: Maximum number of new tokens to generate (default: 256)
    --temperature: Temperature for sampling (default: 0.7)
    --top_p: Top-p sampling parameter (default: 0.9)
    --top_k: Top-k sampling parameter (default: 50)
    --repetition_penalty: Repetition penalty (default: 1.1)
    --quantization: Quantization method (choices: "4bit", "8bit")
    --use_fast_kernels: Use fast attention kernels (flag, default: False)
    --interactive: Run in interactive mode (flag, default: True)
    --length_penalty: Length penalty for generation (default: 0.6)
    --no_repeat_ngram_size: Size of n-grams to prevent repetition (default: 3)

    Raises:
    ------
    Any exceptions that occur during execution are logged and re-raised.

    Notes:
    -----
    This function ensures proper resource cleanup by calling model.stop()
    and model.clear_memory() in a finally block.
    """
    parser = argparse.ArgumentParser(
        description="Run chat completion with Llama 3.1 model."
    )
    parser.add_argument(
        "--model_name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="Name of the tokenizer to use",
    )
    parser.add_argument(
        "--output_dir",
        default="./model/base/meta-llama/Llama-3.2-1B-Instruct",
        help="Directory to load the model from",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Temperature for sampling"
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9, help="Top-p sampling parameter"
    )
    parser.add_argument(
        "--top_k", type=int, default=50, help="Top-k sampling parameter"
    )
    parser.add_argument(
        "--repetition_penalty", type=float, default=1.1, help="Repetition penalty"
    )
    parser.add_argument(
        "--quant", choices=["4bit", "8bit"], help="Quantization method"
    )
    parser.add_argument(
        "--use_fast_kernels",type = bool, default=True, help="Use fast attention kernels"
    )
    parser.add_argument(
        "--interactive", type = bool, default=True, help="Run in interactive mode"
    )
    parser.add_argument(
        "--length_penalty", type=float, default=0.6, help="Length penalty for generation"
    )
    parser.add_argument(
        "--no_repeat_ngram_size", type=int, default=3, help="Size of n-grams to prevent repetition"
    )
    parser.add_argument(
        "--adapter_path",
        help="Path to LoRA adapter weights",
    )


    check_gpu()
    args = parser.parse_args()

    model = Model(args.model_name, args.tokenizer_name, args.output_dir, adapter_path=args.adapter_path)

    try:
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "repetition_penalty": args.repetition_penalty,
            "do_sample": True,
            "length_penalty": args.length_penalty,
            "no_repeat_ngram_size": args.no_repeat_ngram_size,
            "early_stopping": True,
        }

        run_chat(model, generation_kwargs, interactive=args.interactive)

    except (ValueError, TypeError) as e:
        logger.info(f"An error occurred: {e}")
    finally:
        model.stop()
        model.clear_memory()


if __name__ == "__main__":
    main()