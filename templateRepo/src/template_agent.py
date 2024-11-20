from typing import Dict, List, Any, Callable, Optional
from dataclasses import dataclass
import os
from openai import OpenAI
import json
from abc import ABC, abstractmethod

@dataclass
class ToolDefinition:
    """Definition of a tool including its parameters and function."""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Callable
    required_params: List[str]

class BaseToolSet(ABC):
    """Abstract base class for tool sets."""
    
    @abstractmethod
    def get_tools(self) -> Dict[str, ToolDefinition]:
        """Return dictionary of tools available in this tool set."""
        pass

class TemplateAgent:
    """Template-based agent that can be configured with custom tools."""
    
    def __init__(self, 
                 system_prompt: str,
                 tool_sets: List[BaseToolSet],
                 model: str = "gpt-4"):
        """
        Initialize the agent with custom tools and prompt.
        
        Args:
            system_prompt: The system prompt for the agent
            tool_sets: List of tool sets to make available to the agent
            model: OpenAI model to use
        """
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.model = model
        self.tools = {}
        
        # Register all tools from provided tool sets
        for tool_set in tool_sets:
            self.tools.update(tool_set.get_tools())
            
        self.tool_definitions = self._create_tool_definitions()

    def _create_tool_definitions(self) -> List[Dict]:
        """Create tool definitions in OpenAI format."""
        definitions = []
        for tool_name, tool in self.tools.items():
            definitions.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": {
                        "type": "object",
                        "properties": tool.parameters,
                        "required": tool.required_params
                    }
                }
            })
        return definitions

    def execute_tool(self, function_name: str, function_args: Dict[str, Any]) -> Any:
        """Execute a specific tool with given arguments."""
        if function_name not in self.tools:
            raise ValueError(f"Unknown function: {function_name}")
            
        return self.tools[function_name].function(**function_args)

    def process_query(self, user_query: str) -> str:
        """Process user query using tools."""
        try:
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_query}
            ]

            while True:
                # Get response with potential tool calls
                response = self.client.chat.completions.create(
                    model=self.model,
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

# Example implementation of a tool set
class TextAnalysisTools(BaseToolSet):
    """Example tool set for text analysis."""
    
    def get_tools(self) -> Dict[str, ToolDefinition]:
        return {
            "count_letters": ToolDefinition(
                name="count_letters",
                description="Count occurrences of a specific letter in text",
                parameters={
                    "text": {"type": "string", "description": "The text to analyze"},
                    "letter": {"type": "string", "description": "The letter to count"}
                },
                required_params=["text", "letter"],
                function=self._count_letters
            ),
            "count_words": ToolDefinition(
                name="count_words",
                description="Count total number of words in text",
                parameters={
                    "text": {"type": "string", "description": "The text to analyze"}
                },
                required_params=["text"],
                function=self._count_words
            )
        }
    
    def _count_letters(self, text: str, letter: str) -> int:
        return text.lower().count(letter.lower())
    
    def _count_words(self, text: str) -> int:
        return len(text.split())

# Example usage
def main():
    # Define system prompt
    system_prompt = """You are a helpful assistant that analyzes text. 
    You can use multiple tools in sequence to provide comprehensive analysis.
    Feel free to use the same tool multiple times if needed for different parts of the text.
    Respond conversationally but ensure accuracy in your analysis."""
    
    # Create agent with text analysis tools
    agent = TemplateAgent(
        system_prompt=system_prompt,
        tool_sets=[TextAnalysisTools()],
        model="gpt-4"
    )
    
    # Example queries
    example_queries = [
        "How many 's's are in Mississippi?",
        "Count the words in 'The quick brown fox'",
    ]
    
    for query in example_queries:
        print(f"\nQuery: {query}")
        response = agent.process_query(query)
        print(f"Response: {response}")

# Example of creating a custom tool set
class CustomMathTools(BaseToolSet):
    """Example of custom math tools."""
    
    def get_tools(self) -> Dict[str, ToolDefinition]:
        return {
            "calculate_sum": ToolDefinition(
                name="calculate_sum",
                description="Calculate sum of numbers",
                parameters={
                    "numbers": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "List of numbers to sum"
                    }
                },
                required_params=["numbers"],
                function=sum
            )
        }

if __name__ == "__main__":
    main()