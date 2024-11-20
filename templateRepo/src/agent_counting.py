from typing import Dict, List, Any
from dataclasses import dataclass
import os
from openai import OpenAI
import json

@dataclass
class AnalysisTool:
    name: str
    description: str
    parameters: dict

class StringAnalysisTools:
    """Collection of string analysis tools."""
    
    def count_letters(self, text: str, letter: str) -> int:
        """Count occurrences of a specific letter in text."""
        return text.lower().count(letter.lower())
    
    def count_words(self, text: str) -> int:
        """Count total number of words in text."""
        return len(text.split())
    
    def analyze_letter_frequency(self, text: str) -> Dict[str, int]:
        """Analyze frequency of all letters in text."""
        return {char: text.lower().count(char) for char in set(text.lower()) if char.isalpha()}
    
    def find_letter_positions(self, text: str, letter: str) -> List[int]:
        """Find positions of a specific letter in text."""
        return [i for i, char in enumerate(text.lower()) if char == letter.lower()]

class OpenAIAgent:
    """OpenAI-powered agent for text analysis."""
    
    def __init__(self):
        self.api_key = os.getenv("OPEN_AI_API_KEY")
        if not self.api_key:
            raise ValueError("OPEN_AI_API_KEY environment variable not set")
        
        self.client = OpenAI(api_key=self.api_key)
        self.tools = StringAnalysisTools()
        self.tool_definitions = self._define_tools()

    def _define_tools(self) -> List[Dict]:
        """Define available tools for the OpenAI model."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "count_letters",
                    "description": "Count occurrences of a specific letter in text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to analyze"},
                            "letter": {"type": "string", "description": "The letter to count"}
                        },
                        "required": ["text", "letter"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "count_words",
                    "description": "Count total number of words in text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to analyze"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_letter_frequency",
                    "description": "Analyze frequency of all letters in text",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to analyze"}
                        },
                        "required": ["text"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "find_letter_positions",
                    "description": "Find positions of a specific letter in text (0-based index)",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string", "description": "The text to analyze"},
                            "letter": {"type": "string", "description": "The letter to find"}
                        },
                        "required": ["text", "letter"]
                    }
                }
            }
        ]

    def execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """Execute a specific tool with given arguments."""
        tool_mapping = {
            "count_letters": self.tools.count_letters,
            "count_words": self.tools.count_words,
            "analyze_letter_frequency": self.tools.analyze_letter_frequency,
            "find_letter_positions": self.tools.find_letter_positions
        }
        
        if function_name not in tool_mapping:
            raise ValueError(f"Unknown function: {function_name}")
            
        return tool_mapping[function_name](**function_args)

    def process_query(self, user_query: str) -> str:
        """Process user query using OpenAI and string analysis tools."""
        try:
            messages = [
                {"role": "system", "content": """You are a helpful assistant that analyzes text. 
                You can use multiple tools in sequence to provide comprehensive analysis.
                Feel free to use the same tool multiple times if needed for different parts of the text.
                Respond conversationally but ensure accuracy in your analysis."""},
                {"role": "user", "content": user_query}
            ]

            while True:
                # Get response with potential tool calls
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=self.tool_definitions,
                    tool_choice="auto"
                )

                assistant_message = response.choices[0].message
                messages.append(assistant_message)

                # If no tool calls, return the response
                if not hasattr(assistant_message, 'tool_calls') or not assistant_message.tool_calls:
                    return assistant_message.content

                # Process all tool calls in this response
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    
                    try:
                        result = self.execute_tool(function_name, function_args)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": json.dumps(result)
                        })
                    except Exception as e:
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": f"Error executing tool: {str(e)}"
                        })

        except Exception as e:
            return f"An error occurred: {str(e)}"

def main():
    """Example usage with multiple tool calls."""
    try:
        agent = OpenAIAgent()
        
        # Example queries that require multiple tool usage
        example_queries = [
            "How many Rs are in the word Strawberry?",
            "How many words are in 'The quick brown fox'?",
            "Count the letter 's' in Mississippi.",
            "How many 's' in Ross and how many words in 'Innovation and Applied Research Group', then tell me how many 's's and 'i's are in Mississippi?",
            "In the phrase 'hello world', count both the total words and the letter 'l' positions",
            "Analyze the word 'banana': count total letters, find all 'a' positions, and show letter frequency",
            "Compare the number of vowels in 'hello' and 'world'"
        ]
        
        for query in example_queries:
            print(f"\nQuery: {query}")
            response = agent.process_query(query)
            print(f"Response: {response}")
            
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()