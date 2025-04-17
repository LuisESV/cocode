"""
MemoryManager: Manage conversation history and context.

This module provides functionality to save and load conversation history,
manage contexts for different files or tasks, and provide relevant context
to the LLM.
"""

import os
import json
import time
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryManager:
    """Manage conversation history and context."""
    
    def __init__(self, work_dir: str = None):
        """
        Initialize the MemoryManager.
        
        Args:
            work_dir: The working directory (default: current directory)
        """
        self.work_dir = work_dir or os.getcwd()
        
        # Set up storage directory
        self.storage_dir = self._get_storage_dir()
        os.makedirs(self.storage_dir, exist_ok=True)
        
        # Current session data
        self.session_id = self._generate_session_id()
        self.current_context = None
        self.contexts = {}
        
        logger.info(f"MemoryManager initialized with storage directory: {self.storage_dir}")
        
        # Load available contexts
        self._load_available_contexts()
    
    def _get_storage_dir(self) -> str:
        """Get the storage directory for memory files."""
        # Use a .cocode directory inside the working directory
        storage_dir = os.path.join(self.work_dir, ".cocode", "memory")
        return storage_dir
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        timestamp = int(time.time())
        random_part = os.urandom(4).hex()
        return f"{timestamp}-{random_part}"
    
    def _load_available_contexts(self) -> None:
        """Load information about available contexts."""
        try:
            # Look for context files in the storage directory
            for file_path in Path(self.storage_dir).glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Store basic metadata
                    context_id = file_path.stem
                    self.contexts[context_id] = {
                        "id": context_id,
                        "name": data.get("name", "Unnamed context"),
                        "created_at": data.get("created_at"),
                        "updated_at": data.get("updated_at"),
                        "file_path": str(file_path),
                        "message_count": len(data.get("messages", [])),
                    }
                except Exception as e:
                    logger.warning(f"Error loading context from {file_path}: {e}")
            
            logger.info(f"Loaded {len(self.contexts)} contexts from storage")
        except Exception as e:
            logger.error(f"Error loading available contexts: {e}")
    
    def get_available_contexts(self) -> List[Dict[str, Any]]:
        """
        Get a list of available contexts.
        
        Returns:
            List of context metadata
        """
        return list(self.contexts.values())
    
    def create_context(self, name: str, description: str = None, metadata: Dict = None) -> str:
        """
        Create a new context.
        
        Args:
            name: The name of the context
            description: Optional description
            metadata: Optional additional metadata
            
        Returns:
            The ID of the created context
        """
        # Generate a context ID
        context_id = hashlib.md5(f"{name}-{time.time()}".encode()).hexdigest()[:12]
        
        # Create the context data
        now = datetime.now().isoformat()
        context_data = {
            "id": context_id,
            "name": name,
            "description": description or "",
            "created_at": now,
            "updated_at": now,
            "metadata": metadata or {},
            "messages": [],
        }
        
        # Save the context to a file
        file_path = os.path.join(self.storage_dir, f"{context_id}.json")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2)
        
        # Add to in-memory contexts
        self.contexts[context_id] = {
            "id": context_id,
            "name": name,
            "created_at": now,
            "updated_at": now,
            "file_path": file_path,
            "message_count": 0,
        }
        
        # Set as current context
        self.current_context = context_id
        
        logger.info(f"Created new context: {name} (ID: {context_id})")
        return context_id
    
    def get_context(self, context_id: str) -> Optional[Dict]:
        """
        Get a context by ID.
        
        Args:
            context_id: The ID of the context
            
        Returns:
            The context data, or None if not found
        """
        if context_id not in self.contexts:
            logger.warning(f"Context not found: {context_id}")
            return None
        
        context_info = self.contexts[context_id]
        file_path = context_info["file_path"]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading context {context_id}: {e}")
            return None
    
    def set_current_context(self, context_id: str) -> bool:
        """
        Set the current active context.
        
        Args:
            context_id: The ID of the context to set as active
            
        Returns:
            Whether the operation was successful
        """
        if context_id not in self.contexts:
            logger.warning(f"Context not found: {context_id}")
            return False
        
        self.current_context = context_id
        logger.info(f"Set current context: {context_id}")
        return True
    
    def get_current_context_id(self) -> Optional[str]:
        """
        Get the ID of the current context.
        
        Returns:
            The ID of the current context, or None if not set
        """
        return self.current_context
    
    def add_messages_to_context(self, messages: List[Dict], context_id: str = None) -> bool:
        """
        Add messages to a context.
        
        Args:
            messages: The messages to add
            context_id: The ID of the context (uses current if not specified)
            
        Returns:
            Whether the operation was successful
        """
        context_id = context_id or self.current_context
        
        if not context_id:
            logger.warning("No context specified or selected")
            return False
        
        if context_id not in self.contexts:
            logger.warning(f"Context not found: {context_id}")
            return False
        
        try:
            # Load the current context
            context_data = self.get_context(context_id)
            if not context_data:
                return False
            
            # Add the messages
            context_data["messages"].extend(messages)
            
            # Update timestamp
            context_data["updated_at"] = datetime.now().isoformat()
            
            # Save the updated context
            file_path = self.contexts[context_id]["file_path"]
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context_data, f, indent=2)
            
            # Update in-memory metadata
            self.contexts[context_id]["message_count"] = len(context_data["messages"])
            self.contexts[context_id]["updated_at"] = context_data["updated_at"]
            
            logger.info(f"Added {len(messages)} messages to context {context_id}")
            return True
        except Exception as e:
            logger.error(f"Error adding messages to context {context_id}: {e}")
            return False
    
    def get_messages_from_context(self, context_id: str = None, limit: int = None) -> List[Dict]:
        """
        Get messages from a context.
        
        Args:
            context_id: The ID of the context (uses current if not specified)
            limit: Maximum number of messages to return
            
        Returns:
            List of messages
        """
        context_id = context_id or self.current_context
        
        if not context_id:
            logger.warning("No context specified or selected")
            return []
        
        try:
            # Load the context
            context_data = self.get_context(context_id)
            if not context_data:
                return []
            
            # Get the messages
            messages = context_data.get("messages", [])
            
            # Apply limit if specified
            if limit is not None:
                messages = messages[-limit:]
            
            return messages
        except Exception as e:
            logger.error(f"Error getting messages from context {context_id}: {e}")
            return []
    
    def find_context_for_path(self, file_path: str) -> Optional[str]:
        """
        Find a context related to a specific file path.
        
        Args:
            file_path: The file path to find a context for
            
        Returns:
            The ID of the most relevant context, or None if not found
        """
        best_match = None
        best_relevance = 0
        
        # Iterate through available contexts
        for context_id, context_info in self.contexts.items():
            # Load the context
            context_data = self.get_context(context_id)
            if not context_data:
                continue
            
            # Check metadata for file path
            metadata = context_data.get("metadata", {})
            
            # Calculate relevance
            relevance = 0
            
            # Direct match in metadata
            if metadata.get("file_path") == file_path:
                relevance += 10
            
            # Partial match in metadata
            elif file_path in metadata.get("file_path", ""):
                relevance += 5
            
            # Check messages for file path mentions
            for message in context_data.get("messages", []):
                if isinstance(message.get("content"), str) and file_path in message["content"]:
                    relevance += 1
            
            # Update best match if more relevant
            if relevance > best_relevance:
                best_relevance = relevance
                best_match = context_id
        
        return best_match if best_relevance > 0 else None
    
    def save_session(self, messages: List[Dict], metadata: Dict = None) -> str:
        """
        Save the current session to a new context.
        
        Args:
            messages: The messages to save
            metadata: Optional additional metadata
            
        Returns:
            The ID of the created context
        """
        # Create a name for the context
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Try to extract a better name from the first message
        name = f"Session {timestamp}"
        if messages and len(messages) > 0:
            first_message = messages[0]
            if isinstance(first_message.get("content"), str):
                # Use the first few words of the first message
                content = first_message["content"]
                name_part = ' '.join(content.split()[:5])
                if name_part:
                    name = f"{name_part}... ({timestamp})"
        
        # Create a new context
        context_id = self.create_context(
            name=name,
            description=f"Session saved on {timestamp}",
            metadata=metadata or {},
        )
        
        # Add the messages
        self.add_messages_to_context(messages, context_id)
        
        return context_id
    
    def delete_context(self, context_id: str) -> bool:
        """
        Delete a context.
        
        Args:
            context_id: The ID of the context to delete
            
        Returns:
            Whether the operation was successful
        """
        if context_id not in self.contexts:
            logger.warning(f"Context not found: {context_id}")
            return False
        
        try:
            # Delete the file
            file_path = self.contexts[context_id]["file_path"]
            os.remove(file_path)
            
            # Remove from in-memory contexts
            del self.contexts[context_id]
            
            # Reset current context if it was the deleted one
            if self.current_context == context_id:
                self.current_context = None
            
            logger.info(f"Deleted context: {context_id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting context {context_id}: {e}")
            return False