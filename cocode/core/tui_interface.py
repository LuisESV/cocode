"""
TUIApp: Terminal User Interface for CoCoDe.

This module provides a Textual-based Terminal UI for the CoCoDe application.
"""

import os
import sys
import time
import logging
from typing import Dict, List, Optional, Any, Union, Callable

from textual.app import App, ComposeResult
from textual.widgets import Header, Footer, Input, TextLog, Tree, Static, Button, Label
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.binding import Binding
from textual import events
from rich.syntax import Syntax
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.console import RenderableType

# Local imports
from .model_connector import ModelConnector
from .interpreter_wrapper import InterpreterWrapper
from .git_agent import GitAgent
from .file_indexer import FileIndexer
from .memory_manager import MemoryManager

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OutputPanel(Static):
    """Panel for displaying conversation and output."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_focus = True
    
    def add_user_message(self, message: str):
        """Add a user message to the output."""
        self.update(Panel(message, title="User", border_style="blue"))
    
    def add_assistant_message(self, message: str):
        """Add an assistant message to the output."""
        try:
            # Try to render as Markdown
            md = Markdown(message)
            self.update(Panel(md, title="CoCoDe", border_style="green"))
        except Exception as e:
            # Fallback to plain text
            self.update(Panel(message, title="CoCoDe", border_style="green"))
    
    def add_system_message(self, message: str):
        """Add a system message to the output."""
        self.update(Panel(message, title="System", border_style="yellow"))
    
    def add_code_output(self, output: str, language: str = "text"):
        """Add code output to the output panel."""
        syntax = Syntax(output, language, theme="monokai", line_numbers=True)
        self.update(Panel(syntax, title=f"Output ({language})", border_style="cyan"))

class FileTree(Tree):
    """File tree widget for displaying the directory structure."""
    
    def __init__(self, *args, **kwargs):
        self.root_dir = kwargs.pop("root_dir", os.getcwd())
        super().__init__(*args, **kwargs)
        self.root.expand()
        
    def on_mount(self):
        """Called when the widget is mounted."""
        self.build_file_tree()
    
    def build_file_tree(self):
        """Build the file tree from the root directory."""
        self.root.label = os.path.basename(self.root_dir)
        self.root.data = {"path": self.root_dir, "type": "directory"}
        
        # Clear existing nodes
        self.root.remove_children()
        
        # Add files and directories
        self._add_directory_to_tree(self.root_dir, self.root)
    
    def _add_directory_to_tree(self, dir_path: str, parent_node):
        """Add a directory and its contents to the tree."""
        try:
            # Get list of files and directories
            items = os.listdir(dir_path)
            
            # Skip certain directories
            if os.path.basename(dir_path) in ['.git', 'node_modules', '__pycache__']:
                return
            
            # Add directories first
            for item in sorted(items):
                item_path = os.path.join(dir_path, item)
                
                # Skip hidden files/directories
                if item.startswith('.'):
                    continue
                
                # Add to tree
                if os.path.isdir(item_path):
                    node = parent_node.add(
                        item, 
                        expand=False, 
                        data={"path": item_path, "type": "directory"}
                    )
                    self._add_directory_to_tree(item_path, node)
            
            # Then add files
            for item in sorted(items):
                item_path = os.path.join(dir_path, item)
                
                # Skip hidden files/directories
                if item.startswith('.'):
                    continue
                
                # Add to tree
                if os.path.isfile(item_path):
                    parent_node.add(
                        item, 
                        data={"path": item_path, "type": "file"}
                    )
        except Exception as e:
            logger.error(f"Error adding directory to tree: {e}")
            # Add an error node
            parent_node.add(f"Error: {str(e)}")
    
    def on_tree_node_selected(self, event):
        """Called when a tree node is selected."""
        node = event.node
        if not node.data:
            return
        
        # Post a message to the app
        self.post_message(FileSelectedMessage(node.data))

class FileSelectedMessage(events.Message):
    """Message sent when a file is selected in the file tree."""
    
    def __init__(self, file_data: Dict[str, Any]):
        self.file_data = file_data
        super().__init__()

class ConfirmDialog(Container):
    """Dialog for confirming actions."""
    
    def __init__(self, message: str, detail: str = None):
        super().__init__()
        self.message = message
        self.detail = detail
        self.callback = None
    
    def compose(self) -> ComposeResult:
        yield Container(
            Label(self.message),
            Static(self.detail) if self.detail else None,
            Horizontal(
                Button("Confirm", variant="primary", id="confirm"),
                Button("Cancel", variant="error", id="cancel"),
                classes="buttons"
            ),
            classes="dialog"
        )
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Called when a button is pressed."""
        if event.button.id == "confirm":
            if self.callback:
                self.callback(True)
        else:
            if self.callback:
                self.callback(False)
        
        # Remove the dialog
        self.remove()
    
    def set_callback(self, callback: Callable[[bool], None]):
        """Set the callback to be called when the dialog is closed."""
        self.callback = callback

class TUIApp(App):
    """Terminal User Interface for CoCoDe."""
    
    TITLE = "CoCoDe - AI Coding Assistant"
    CSS_PATH = None  # Add a CSS file path here if needed
    
    # Reactive variables
    current_file = reactive(None)
    
    def __init__(self, 
                 interpreter: InterpreterWrapper,
                 git_agent: GitAgent,
                 file_indexer: FileIndexer,
                 memory_manager: MemoryManager,
                 model_connector: ModelConnector,
                 work_dir: str = None):
        """
        Initialize the TUI application.
        
        Args:
            interpreter: The InterpreterWrapper instance
            git_agent: The GitAgent instance
            file_indexer: The FileIndexer instance
            memory_manager: The MemoryManager instance
            model_connector: The ModelConnector instance
            work_dir: The working directory (default: current directory)
        """
        super().__init__()
        
        self.interpreter = interpreter
        self.git_agent = git_agent
        self.file_indexer = file_indexer
        self.memory_manager = memory_manager
        self.model_connector = model_connector
        self.work_dir = work_dir or os.getcwd()
        
        # Set file edit confirmation callback
        self.interpreter.set_confirm_edit_callback(self.confirm_file_edit)
        self.git_agent.set_confirm_callback(self.confirm_git_operation)
        
        logger.info("TUIApp initialized")
    
    def compose(self) -> ComposeResult:
        """Compose the UI layout."""
        # Header with app title and status
        yield Header()
        
        # Main content with file tree, output, and input
        with Container(id="main"):
            # Left panel with file tree
            with Container(id="left-panel"):
                yield Label("Files", id="files-label")
                yield FileTree(id="file-tree", root_dir=self.work_dir)
            
            # Right panel with output and input
            with Container(id="right-panel"):
                yield OutputPanel(id="output")
                yield Input(placeholder="Enter your message...", id="input")
        
        # Footer with help text and status
        yield Footer()
    
    def on_mount(self):
        """Called when the app is mounted."""
        # Add a welcome message
        self.query_one("#output", OutputPanel).add_system_message(
            "Welcome to CoCoDe!\n\n"
            "Enter your request below to get started. You can ask for help with coding tasks, "
            "explore the codebase, make changes to files, and more."
        )
        
        # Refresh the file tree
        self.query_one("#file-tree", FileTree).build_file_tree()
        
        # Focus the input
        self.query_one("#input", Input).focus()
    
    def on_input_submitted(self, event: Input.Submitted):
        """Called when the input is submitted."""
        input_widget = self.query_one("#input", Input)
        user_input = input_widget.value
        
        if not user_input.strip():
            return
        
        # Clear the input
        input_widget.value = ""
        
        # Add user message to output
        output_widget = self.query_one("#output", OutputPanel)
        output_widget.add_user_message(user_input)
        
        # Process the input
        self.process_user_input(user_input)
    
    def on_file_selected_message(self, message: FileSelectedMessage):
        """Called when a file is selected in the file tree."""
        file_data = message.file_data
        
        # Set the current file
        self.current_file = file_data
        
        # If it's a file, show its content
        if file_data["type"] == "file":
            try:
                with open(file_data["path"], 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Determine language based on file extension
                _, ext = os.path.splitext(file_data["path"])
                language = self._get_language_for_extension(ext)
                
                # Show file content
                self.query_one("#output", OutputPanel).add_code_output(
                    content, language=language
                )
            except Exception as e:
                self.query_one("#output", OutputPanel).add_system_message(
                    f"Error reading file: {str(e)}"
                )
    
    def _get_language_for_extension(self, extension: str) -> str:
        """Get the language name for a file extension."""
        extension = extension.lower()
        
        # Map of extensions to languages
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.jsx': 'jsx',
            '.ts': 'typescript',
            '.tsx': 'tsx',
            '.html': 'html',
            '.css': 'css',
            '.json': 'json',
            '.md': 'markdown',
            '.yml': 'yaml',
            '.yaml': 'yaml',
            '.java': 'java',
            '.c': 'c',
            '.cpp': 'cpp',
            '.h': 'c',
            '.hpp': 'cpp',
            '.rs': 'rust',
            '.go': 'go',
            '.rb': 'ruby',
            '.php': 'php',
        }
        
        return extension_map.get(extension, 'text')
    
    def process_user_input(self, user_input: str):
        """Process user input and get a response."""
        try:
            # Update status
            self.query_one(Footer).highlight = True
            
            # Get response from the interpreter
            output_widget = self.query_one("#output", OutputPanel)
            
            # Start building response
            self.query_one(Footer).update("Thinking...")
            
            # Process the response chunks
            response_chunks = []
            
            # Get chunks from interpreter
            for chunk in self.interpreter.chat(user_input):
                # Process the chunk based on its type
                if chunk["type"] == "text":
                    response_chunks.append(chunk["content"])
                    
                    # Update in real-time
                    output_widget.add_assistant_message(''.join(response_chunks))
                
                elif chunk["type"] == "tool_call":
                    # Handle tool calls (e.g., file operations)
                    pass
                
                elif chunk["type"] == "console":
                    # Handle console output
                    output_widget.add_code_output(chunk["content"])
                
                elif chunk["type"] == "error":
                    # Handle errors
                    output_widget.add_system_message(f"Error: {chunk['content']}")
            
            # Update footer status
            self.query_one(Footer).highlight = False
            self.query_one(Footer).update("")
        except Exception as e:
            logger.error(f"Error processing user input: {e}")
            output_widget = self.query_one("#output", OutputPanel)
            output_widget.add_system_message(f"Error: {str(e)}")
            
            # Update footer status
            self.query_one(Footer).highlight = False
            self.query_one(Footer).update("")
    
    def confirm_file_edit(self, message: str, diff: str) -> bool:
        """
        Confirm a file edit operation.
        
        Args:
            message: The message to display
            diff: The diff to show
            
        Returns:
            Whether the edit was confirmed
        """
        # Create a dialog
        dialog = ConfirmDialog(message, diff)
        
        # Show the dialog
        self.mount(dialog)
        
        # Create a future to get the result
        confirmed = [None]
        
        def callback(result):
            confirmed[0] = result
        
        dialog.set_callback(callback)
        
        # Wait for the result
        while confirmed[0] is None:
            time.sleep(0.1)
        
        return confirmed[0]
    
    def confirm_git_operation(self, message: str, detail: str = None) -> bool:
        """
        Confirm a Git operation.
        
        Args:
            message: The message to display
            detail: The detail to show
            
        Returns:
            Whether the operation was confirmed
        """
        # Create a dialog
        dialog = ConfirmDialog(message, detail)
        
        # Show the dialog
        self.mount(dialog)
        
        # Create a future to get the result
        confirmed = [None]
        
        def callback(result):
            confirmed[0] = result
        
        dialog.set_callback(callback)
        
        # Wait for the result
        while confirmed[0] is None:
            time.sleep(0.1)
        
        return confirmed[0]