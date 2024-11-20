"""Module provides functionality for the local model, including logging and date handling."""

import os
from pathlib import Path
from typing import Any

import torch
from logger import Logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

## TODO: Add token count output/performance metrics as a method here


class Model:
    """
    A class to represent a large language model.

    Attributes
    ----------
        model_name (str): The name of the model.
        tokenizer_name (str): The name of the tokenizer.
        output_dir (str): The directory where model artifacts are saved.

    """

    def __init__(self, model_name: str, tokenizer_name: str, output_dir: str, adapter_path: str | None = None) -> None:
        """
        Initialize the Model class.

        Args:
        ----
            model_name (str): Name of the model to use.
            tokenizer_name (str): Name of the tokenizer to use.
            output_dir (str): Directory to save/load the model and related artifacts.

        Attributes:
        ----------
            model_name (str): Name of the model.
            tokenizer_name (str): Name of the tokenizer.
            output_dir (str): Directory for model storage.
            model: The loaded model (initially None).
            tokenizer: The loaded tokenizer (initially None).
            device (torch.device): Device to use for computations.
            logger (logging.Logger): Logger for the class.
            adapter_path (str | None): Path to the adapter model.
            hf_token (str): Hugging Face access token.

        """
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.output_dir = output_dir
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter_path = adapter_path
        self.setup_logging()
        self.check_hf_token()
        self.load_model()

    ### Logging and support functions ###
    def setup_logging(self) -> None:
        """
        Set up logging for the Model class.

        Uses the custom Logger class.
        """
        self.logger = Logger(initial_level="DEBUG", module_name="Local Model")

    def check_hf_token(self) -> None:
        """
        Check for the existence of the HF_TOKEN environment variable.

        Raises
        ------
            Error: If HF_TOKEN environment variable is not set.

        """
        self.hf_token = "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"
        if not self.hf_token:
            self.logger.error("HF_TOKEN environment variable not set. Please set up a Hugging Face access token with at least read-only access.")
            self.logger.info("For more information, visit: https://huggingface.co/docs/hub/en/security-tokens")
            raise OSError

    ### Model loading and handling ###
    def download_model(self) -> None:
        """
        Download the model and tokenizer.

        Creates the output directory if it doesn't exist and saves the pretrained model and tokenizer.
        """
        if not Path(self.output_dir).exists():
            Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        AutoModelForCausalLM.from_pretrained(self.model_name).save_pretrained(self.output_dir)
        AutoTokenizer.from_pretrained(self.tokenizer_name).save_pretrained(self.output_dir)

    def load_model(self, quantization: str | None = None, use_fast_kernels: bool = False) -> None:
        """
        Load the model and tokenizer, optionally with LoRA adapter.

        Args:
        ----
            quantization (str | None): Quantization method to use ('4bit' or '8bit'). Default is None.
            use_fast_kernels (bool): Whether to use fast attention kernels. Default is False.

        Raises:
        ------
            Exception: If there's an error loading the model.
        """
        self.logger.info(f"Checking for model {self.model_name} in {self.output_dir}")

        if not Path(self.output_dir).exists():
            self.logger.info(f"Model not found locally. Downloading model {self.model_name}")
            self.download_model()

        try:
            self.logger.info(f"Loading model {self.model_name} from {self.output_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name, 
                add_eos_token=True, 
                add_bos_token=True
            )

            if self.tokenizer.pad_token is None:
                self.logger.debug("Tokenizer has no pad token. Setting pad token to eos token.")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

            model_kwargs = {
                "device_map": self.device,
                "torch_dtype": torch.float16,
                "low_cpu_mem_usage": True,
            }

            if quantization == "4bit":
                model_kwargs["load_in_4bit"] = True
            elif quantization == "8bit":
                model_kwargs["load_in_8bit"] = True

            if use_fast_kernels:
                model_kwargs["use_flash_attention_2"] = True

            self.model = AutoModelForCausalLM.from_pretrained(
                self.output_dir, 
                **model_kwargs
            )

            if self.model.config.pad_token_id is None:
                self.logger.debug("Model has no pad token. Setting pad token to eos token.")
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            self.logger.info("Model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

 
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs: dict[str, Any]) -> str:
        """Generate a response based on the given prompt."""
        if self.model is None or self.tokenizer is None:
            error_msg = "Model or tokenizer not loaded. Call load_model() first."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        self.logger.info("Generating response")
        try:
            # Tokenize input
            tokenized_input = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True
            )
            
            # Move to GPU if available
            if torch.cuda.is_available():
                tokenized_input = {k: v.cuda() for k, v in tokenized_input.items()}
            
            input_length = tokenized_input['input_ids'].shape[1]

            # Base generation config
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "do_sample": True,
                "early_stopping": True,
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
            }
            
            # Update with any additional kwargs
            generation_kwargs.update(kwargs)

            # Generate
            with torch.no_grad():
                generation_kwargs["attention_mask"] = tokenized_input['attention_mask']
                outputs = self.model.generate(
                    tokenized_input['input_ids'],
                    **generation_kwargs
                )

            # Decode output
            generated_tokens = outputs[0, input_length:]
            return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        except Exception as e:
            self.logger.error(f"Exception during generation: {str(e)}")
            raise

    def chat_completion(self, dialogs: list[dict[str, str]], max_new_tokens: int = 2048, **kwargs: dict[str, Any]) -> tuple[list[dict[str, str]], str]:
        """
        Perform chat completion based on the given dialog history.

        Args:
        ----
            dialogs (list[dict[str, str]]): List of dialog messages, each a dict with 'role' and 'content' keys.
            max_new_tokens (int): Maximum number of new tokens to generate, by default 2048.
            **kwargs: Additional keyword arguments for generation.

        Returns:
        -------
            tuple[list[dict[str, str]], str]: The updated dialog history and the generated response.

        Raises:
        ------
            Exception: If there's an error during chat completion.
        """
        self.logger.info("Performing chat completion")
        try:
            # For standard models using HuggingFace chat templates
            chat_prompt = self.tokenizer.apply_chat_template(
                dialogs,
                tokenize=False,
                add_generation_prompt=False
            )
            
            self.logger.debug(f"Generated prompt: {chat_prompt}")

            generated_response = self.generate(
                chat_prompt, 
                max_new_tokens=max_new_tokens, 
                **kwargs
            )

            # For fine-tuned models, we need to clean up the response


            assistant_response = generated_response

            # Add the assistant's response to the dialog history
            dialogs.append({"role": "assistant", "content": assistant_response})
            return dialogs, assistant_response

        except Exception as e:
            self.logger.exception(f"Error during chat completion: {str(e)}")
            raise

    ### Model unloading and memory cleanup ### 
    def stop(self) -> None:
        """
        Stop the model and clear it from memory.

        Sets the model and tokenizer to None and clears CUDA cache if available.
        """
        self.logger.info("Stopping model")
        self.model = None
        self.tokenizer = None
        torch.cuda.empty_cache()

    def clear_memory(self) -> None:
        """
        Clear GPU memory.

        Empties the CUDA cache if a GPU is available.
        """
        self.logger.info("Clearing GPU memory")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()