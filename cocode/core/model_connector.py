"""
ModelConnector: Interface for GPT-4.1 interactions.

This module provides the interface between the application and GPT-4.1,
handling API calls, streaming, token counting, and message management.
"""

import os
import json
import time
import logging
import tiktoken
from typing import Dict, List, Generator, Optional, Union, Any, Callable
from pydantic import BaseModel, Field

import openai
from openai import OpenAI

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MessageContent(BaseModel):
    """Content of a message, supporting text and tool calls/responses."""
    text: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None
    tool_response: Optional[Any] = None


class Message(BaseModel):
    """A message in the conversation."""
    role: str
    content: Union[str, List[MessageContent], None] = None


class ChatCompletionChunk(BaseModel):
    """A chunk of a streaming chat completion."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time())}")
    model: str
    choices: List[Dict]
    usage: Optional[Dict] = None


class ModelConnector:
    """Interface for GPT-4.1 API interactions."""
    
    def __init__(self, 
                 api_key: str = None, 
                 model: str = "gpt-4o", 
                 temperature: float = 0.2,
                 max_tokens: int = None,
                 system_message: str = None):
        """
        Initialize the ModelConnector with the given settings.
        
        Args:
            api_key: OpenAI API key (will use OPENAI_API_KEY env var if not provided)
            model: The model to use (default: gpt-4o)
            temperature: Sampling temperature (default: 0.2)
            max_tokens: Maximum tokens to generate (default: None)
            system_message: Custom system message (default: None)
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Load system message from file if not provided
        if system_message is None:
            try:
                script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                system_prompt_path = os.path.join(script_dir, "system_prompt.txt")
                with open(system_prompt_path, "r") as f:
                    self.system_message = f.read()
            except (FileNotFoundError, IOError):
                logger.warning("System prompt file not found. Using default prompt.")
                self.system_message = "You are CoCoDe, a helpful AI coding assistant."
        else:
            self.system_message = system_message
        
        # Initialize token counter based on model
        self.encoding = self._get_encoding()
        
        # Stats tracking
        self.token_counts = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        
        # Conversation history
        self.messages = []
        self._add_system_message()
        
        logger.info(f"Initialized ModelConnector using model: {self.model}")
    
    def _add_system_message(self):
        """Add the system message to the conversation history."""
        self.messages = [{"role": "system", "content": self.system_message}]
    
    def _get_encoding(self) -> Any:
        """Get the appropriate token encoding for the model."""
        try:
            if "gpt-4" in self.model or "gpt-3.5" in self.model:
                return tiktoken.encoding_for_model(self.model)
            else:
                return tiktoken.get_encoding("cl100k_base")  # Default for new models
        except Exception as e:
            logger.warning(f"Failed to get encoding for {self.model}: {e}")
            return tiktoken.get_encoding("cl100k_base")  # Fallback encoding
            
    def count_tokens(self, text: str) -> int:
        """Count tokens in a string."""
        if not text:
            return 0
        try:
            return len(self.encoding.encode(text))
        except Exception as e:
            logger.warning(f"Error counting tokens: {e}")
            # Fallback: rough approximation
            return len(text) // 4
    
    def count_message_tokens(self, message: Dict) -> int:
        """Count tokens in a message."""
        if not message:
            return 0
        
        # Estimate base tokens for message metadata
        # Each message has ~4 tokens of overhead
        token_count = 4
        
        # Count tokens in message content
        if isinstance(message.get("content"), str):
            token_count += self.count_tokens(message["content"])
        elif isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if isinstance(content_item, dict):
                    if content_item.get("type") == "text":
                        token_count += self.count_tokens(content_item.get("text", ""))
                    # Add handling for other content types as needed
        
        # Add tokens for tool calls if present
        tool_calls = message.get("tool_calls", [])
        if tool_calls:
            for tool_call in tool_calls:
                # Function name and arguments
                token_count += self.count_tokens(tool_call.get("function", {}).get("name", ""))
                token_count += self.count_tokens(json.dumps(tool_call.get("function", {}).get("arguments", {})))
        
        return token_count
    
    def count_conversation_tokens(self) -> int:
        """Count tokens in the entire conversation history."""
        total = 0
        for msg in self.messages:
            total += self.count_message_tokens(msg)
        return total
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation history."""
        self.messages.append({"role": role, "content": content})
    
    def clear_conversation(self) -> None:
        """Clear the conversation history, keeping the system message."""
        self._add_system_message()
    
    def get_conversation_history(self) -> List[Dict]:
        """Get the conversation history."""
        return self.messages
    
    def set_conversation_history(self, messages: List[Dict]) -> None:
        """Set the conversation history."""
        # Ensure the first message is a system message
        if not messages or messages[0].get("role") != "system":
            self._add_system_message()
            if messages:
                self.messages.extend(messages)
        else:
            self.messages = messages
    
    def _prepare_messages(self, user_message: str = None) -> List[Dict]:
        """Prepare messages for API call, optionally adding a new user message."""
        messages = self.messages.copy()
        if user_message:
            messages.append({"role": "user", "content": user_message})
        return messages
    
    def chat(self, 
             user_message: str, 
             stream: bool = True,
             temperature: float = None,
             max_tokens: int = None,
             functions: List[Dict] = None) -> Union[Dict, Generator]:
        """
        Send a message to the model and get a response.
        
        Args:
            user_message: The user's message
            stream: Whether to stream the response (default: True)
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            functions: Function definitions for function calling
            
        Returns:
            If stream=True, returns a generator yielding response chunks
            If stream=False, returns the complete response
        """
        # Add user message to conversation
        self.add_message("user", user_message)
        
        # Prepare parameters
        params = {
            "model": self.model,
            "messages": self.messages,
            "temperature": temperature if temperature is not None else self.temperature,
            "stream": stream,
        }
        
        if max_tokens is not None or self.max_tokens is not None:
            params["max_tokens"] = max_tokens or self.max_tokens
            
        if functions:
            params["tools"] = [
                {"type": "function", "function": func} for func in functions
            ]
        
        logger.info(f"Sending chat request to {self.model}")
        
        try:
            # Call the API
            if stream:
                return self._stream_chat_completion(**params)
            else:
                response = self.client.chat.completions.create(**params)
                self._update_token_counts(response)
                
                # Add assistant response to conversation
                assistant_message = {
                    "role": "assistant", 
                    "content": response.choices[0].message.content
                }
                
                # Handle function calls
                if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                    assistant_message["tool_calls"] = response.choices[0].message.tool_calls
                
                self.messages.append(assistant_message)
                return response
                
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
            
    def _stream_chat_completion(self, **params) -> Generator:
        """Stream the chat completion and handle token counting."""
        try:
            response = self.client.chat.completions.create(**params)
            
            # Track accumulated response for token counting and conversation history
            accumulated_response = {
                "role": "assistant",
                "content": "",
                "tool_calls": []
            }
            
            current_tool_calls = {}
            
            # Process streaming response
            for chunk in response:
                # Extract content from delta
                if hasattr(chunk.choices[0].delta, "content") and chunk.choices[0].delta.content:
                    accumulated_response["content"] += chunk.choices[0].delta.content
                
                # Handle tool calls
                if hasattr(chunk.choices[0].delta, "tool_calls") and chunk.choices[0].delta.tool_calls:
                    for tool_call in chunk.choices[0].delta.tool_calls:
                        # Get the tool call ID
                        tool_call_id = tool_call.id
                        
                        # Initialize if this is a new tool call
                        if tool_call_id not in current_tool_calls:
                            current_tool_calls[tool_call_id] = {
                                "id": tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": "",
                                    "arguments": ""
                                }
                            }
                        
                        # Update function name if present
                        if hasattr(tool_call, "function") and hasattr(tool_call.function, "name"):
                            current_tool_calls[tool_call_id]["function"]["name"] += tool_call.function.name
                        
                        # Update arguments if present
                        if hasattr(tool_call, "function") and hasattr(tool_call.function, "arguments"):
                            current_tool_calls[tool_call_id]["function"]["arguments"] += tool_call.function.arguments
                
                # Yield the chunk for streaming
                yield chunk
            
            # Add accumulated tool calls to response
            if current_tool_calls:
                accumulated_response["tool_calls"] = list(current_tool_calls.values())
            
            # Add assistant response to conversation
            if accumulated_response["content"] or accumulated_response["tool_calls"]:
                if not accumulated_response["tool_calls"]:
                    del accumulated_response["tool_calls"]
                self.messages.append(accumulated_response)
            
            # Estimate token usage
            estimated_prompt_tokens = self.count_conversation_tokens() - self.count_message_tokens(accumulated_response)
            estimated_completion_tokens = self.count_message_tokens(accumulated_response)
            
            self.token_counts["prompt_tokens"] += estimated_prompt_tokens
            self.token_counts["completion_tokens"] += estimated_completion_tokens
            self.token_counts["total_tokens"] += estimated_prompt_tokens + estimated_completion_tokens
            
        except Exception as e:
            logger.error(f"Error in stream chat completion: {e}")
            raise
        
    def _update_token_counts(self, response) -> None:
        """Update token counts from a response."""
        if hasattr(response, "usage") and response.usage:
            self.token_counts["prompt_tokens"] += response.usage.prompt_tokens
            self.token_counts["completion_tokens"] += response.usage.completion_tokens
            self.token_counts["total_tokens"] += response.usage.total_tokens
    
    def handle_tool_call(self, tool_call_id: str, tool_response: Any) -> None:
        """Handle a tool call response and add it to the conversation."""
        self.messages.append({
            "role": "tool",
            "tool_call_id": tool_call_id,
            "content": str(tool_response)
        })
    
    def get_token_counts(self) -> Dict[str, int]:
        """Get the token counts for the session."""
        return self.token_counts