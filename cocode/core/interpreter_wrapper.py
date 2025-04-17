"""
InterpreterWrapper: Interface with Open Interpreter for code execution.

This module adapts Open Interpreter to use our custom ModelConnector
and adds hooks for Git integration and file indexing.
"""

import os
import sys
import logging
import tempfile
from typing import Dict, List, Any, Optional, Generator, Callable

# Open Interpreter imports
import interpreter
from interpreter.core.core import OpenInterpreter
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# Local imports
from .model_connector import ModelConnector
from .git_agent import GitAgent
from .file_indexer import FileIndexer
from .memory_manager import MemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class InterpreterWrapper:
    """Wrapper for Open Interpreter, with custom model connector and extended functionality."""
    
    def __init__(self, 
                 model_connector: ModelConnector,
                 git_agent: GitAgent,
                 file_indexer: FileIndexer,
                 memory_manager: MemoryManager,
                 auto_run: bool = False,
                 verbose: bool = False):
        """
        Initialize the InterpreterWrapper with custom components.
        
        Args:
            model_connector: The ModelConnector instance
            git_agent: The GitAgent instance
            file_indexer: The FileIndexer instance
            memory_manager: The MemoryManager instance
            auto_run: Whether to auto-run code without confirmation
            verbose: Whether to show verbose output
        """
        self.model_connector = model_connector
        self.git_agent = git_agent
        self.file_indexer = file_indexer
        self.memory_manager = memory_manager
        self.console = Console()
        
        # Initialize Open Interpreter
        self.interpreter = OpenInterpreter(
            auto_run=auto_run,
            verbose=verbose,
            offline=False,
        )
        
        # Register our custom LLM
        self._register_custom_llm()
        
        # Initialize messages
        self.messages = []
        self.last_message_count = 0
        
        # Set up file edit confirmation callback
        self.confirm_file_edit_callback = None
        
        logger.info("InterpreterWrapper initialized")
    
    def _register_custom_llm(self):
        """Register our custom ModelConnector with Open Interpreter."""
        # Create a custom LLM adapter class inside Open Interpreter
        class CustomLLMAdapter:
            def __init__(self, model_connector):
                self.model_connector = model_connector
                self.model = model_connector.model
                self.temperature = model_connector.temperature
                self.supports_functions = True
                self.supports_vision = True
                self.context_window = None
                self.max_tokens = model_connector.max_tokens
                
            def chat(self, messages):
                """Convert Open Interpreter messages to our format and send to our ModelConnector."""
                # Convert messages to the format expected by our ModelConnector
                history = self.model_connector.get_conversation_history()
                
                # We need to preserve the system message from our model connector
                system_message = history[0] if history and history[0]["role"] == "system" else None
                
                # Replace the conversation history with the one from Open Interpreter
                # but keep our system message
                new_history = [system_message] if system_message else []
                new_history.extend(messages)
                
                # Set the conversation history in our model connector
                self.model_connector.set_conversation_history(new_history)
                
                # Get a response from our model connector
                # The last message should be from the user
                last_message = messages[-1]
                
                if last_message["role"] == "user":
                    # Use a non-streaming call for compatibility with Open Interpreter
                    response = self.model_connector.chat(
                        user_message=last_message["content"], 
                        stream=False
                    )
                    
                    # Extract the response in the format expected by Open Interpreter
                    choice = response.choices[0]
                    result = {
                        "role": "assistant",
                        "content": choice.message.content,
                    }
                    
                    # Handle function calls
                    if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
                        result["function_call"] = {
                            "name": choice.message.tool_calls[0].function.name,
                            "arguments": choice.message.tool_calls[0].function.arguments
                        }
                    
                    return result
                else:
                    logger.warning(f"Expected user message, got {last_message['role']}")
                    return {"role": "assistant", "content": "I'm sorry, I encountered an error processing your message."}
        
        # Set our custom LLM in the interpreter
        self.interpreter.llm = CustomLLMAdapter(self.model_connector)
        logger.info(f"Registered custom LLM with Open Interpreter: {self.model_connector.model}")
    
    def set_confirm_edit_callback(self, callback: Callable):
        """Set the callback function for confirming file edits."""
        self.confirm_file_edit_callback = callback
    
    def get_file_content(self, file_path: str) -> str:
        """Get the content of a file."""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return f"Error reading file: {str(e)}"
    
    def edit_file(self, file_path: str, content: str = None, old_string: str = None, new_string: str = None) -> bool:
        """
        Edit a file with confirmation.
        
        Args:
            file_path: The path to the file to edit
            content: The new content of the file (for complete rewrites)
            old_string: The string to replace
            new_string: The string to replace with
            
        Returns:
            bool: Whether the edit was successful
        """
        # Validate arguments
        if content is None and (old_string is None or new_string is None):
            logger.error("Must provide either content or both old_string and new_string")
            return False
        
        # Check if file exists
        file_exists = os.path.isfile(file_path)
        
        # Get current content if file exists
        original_content = self.get_file_content(file_path) if file_exists else ""
        
        # Prepare the new content
        if content is not None:
            new_content = content
            # Preview the diff
            diff = self._generate_diff(original_content, new_content, file_path)
        elif old_string is not None and new_string is not None:
            if old_string not in original_content:
                logger.error(f"String to replace not found in {file_path}")
                return False
            new_content = original_content.replace(old_string, new_string)
            # Preview the diff
            diff = self._generate_diff(original_content, new_content, file_path)
        else:
            logger.error("Invalid arguments for edit_file")
            return False
        
        # Ask for confirmation if callback is set
        if self.confirm_file_edit_callback:
            message = f"{'Create' if not file_exists else 'Edit'} file: {file_path}"
            confirmed = self.confirm_file_edit_callback(message, diff)
            if not confirmed:
                logger.info(f"File edit cancelled by user: {file_path}")
                return False
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Write the file
        try:
            with open(file_path, 'w') as f:
                f.write(new_content)
            logger.info(f"Successfully {'created' if not file_exists else 'edited'} file: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error writing file {file_path}: {e}")
            return False
    
    def _generate_diff(self, original: str, new: str, file_path: str) -> str:
        """Generate a diff between original and new content."""
        import difflib
        
        if not original and new:
            # New file
            return f"New file: {file_path}\n\n" + new
        
        # Generate diff
        diff_lines = difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            n=3
        )
        
        return ''.join(diff_lines)
    
    def execute(self, code: str, language: str = "python") -> Dict:
        """
        Execute code using Open Interpreter.
        
        Args:
            code: The code to execute
            language: The language of the code
            
        Returns:
            Dict: The result of the execution
        """
        # Create a new temporary file for the code
        with tempfile.NamedTemporaryFile(suffix=f".{language}", delete=False) as temp:
            temp.write(code.encode())
            temp_path = temp.name
        
        try:
            # Create a message for Open Interpreter
            message = {
                "role": "user",
                "content": f"```{language}\n{code}\n```"
            }
            
            # Execute the code
            self.interpreter.messages.append(message)
            self.last_message_count = len(self.interpreter.messages)
            
            # Use Open Interpreter's chat function to execute the code
            result = self.interpreter.chat()
            
            # Process and return the result
            output = ""
            for msg in result:
                if msg["role"] == "computer" and msg["type"] == "console":
                    output += msg["content"] + "\n"
            
            return {
                "status": "success",
                "output": output
            }
            
        except Exception as e:
            logger.error(f"Error executing code: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
        finally:
            # Clean up the temporary file
            try:
                os.unlink(temp_path)
            except:
                pass
    
    def chat(self, user_input: str, stream: bool = True) -> Generator:
        """
        Process user input and get a response.
        
        Args:
            user_input: The user's input
            stream: Whether to stream the response
            
        Returns:
            Generator: A generator yielding response chunks
        """
        # Check if input is a code block
        if user_input.startswith("```") and user_input.endswith("```"):
            # Extract language and code
            parts = user_input.split("\n", 1)
            if len(parts) > 1:
                language = parts[0].strip("`").strip()
                code = parts[1].rstrip("`").strip()
                
                # Execute the code directly
                result = self.execute(code, language)
                
                # Format the result
                if result["status"] == "success":
                    yield {
                        "type": "console",
                        "content": result["output"]
                    }
                else:
                    yield {
                        "type": "error",
                        "content": result["error"]
                    }
                
                return
        
        # Not a code block, send to the model
        # Add context from file indexer if available
        indexed_context = self.file_indexer.get_context_for_query(user_input)
        if indexed_context:
            enhanced_input = f"{user_input}\n\nContext from codebase:\n{indexed_context}"
        else:
            enhanced_input = user_input
        
        # Send to model and process response
        try:
            for chunk in self.model_connector.chat(enhanced_input, stream=stream):
                # Process the chunk
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    yield {
                        "type": "text",
                        "content": chunk.choices[0].delta.content
                    }
                
                # Handle tool calls
                if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        yield {
                            "type": "tool_call",
                            "tool_call": tool_call
                        }
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            yield {
                "type": "error",
                "content": f"Error: {str(e)}"
            }
    
    def get_conversation_history(self):
        """Get the conversation history."""
        return self.model_connector.get_conversation_history()
    
    def clear_conversation(self):
        """Clear the conversation history."""
        self.model_connector.clear_conversation()
        self.interpreter.messages = []
        
    def get_token_counts(self):
        """Get the token counts for the session."""
        return self.model_connector.get_token_counts()