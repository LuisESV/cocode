#!/usr/bin/env python3
"""
CoCoDe: Code-focused AI assistant powered by GPT-4.1

A terminal UI for OpenAI's GPT-4.1 that helps developers with coding tasks.
Built on Open Interpreter, with Git integration and codebase understanding.
"""

import os
import sys
import typer
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Import core components
from core.model_connector import ModelConnector
from core.interpreter_wrapper import InterpreterWrapper
from core.git_agent import GitAgent
from core.file_indexer import FileIndexer
from core.memory_manager import MemoryManager
from core.tui_interface import TUIApp

app = typer.Typer(help="CoCoDe - A coding assistant powered by GPT-4.1")
console = Console()

# ASCII Art Banner
BANNER = """
 ██████╗ ██████╗  ██████╗ ██████╗ ██████╗ ███████╗
██╔════╝██╔═══██╗██╔════╝██╔═══██╗██╔══██╗██╔════╝
██║     ██║   ██║██║     ██║   ██║██║  ██║█████╗  
██║     ██║   ██║██║     ██║   ██║██║  ██║██╔══╝  
╚██████╗╚██████╔╝╚██████╗╚██████╔╝██████╔╝███████╗
 ╚═════╝ ╚═════╝  ╚═════╝ ╚═════╝ ╚═════╝ ╚══════╝
                                              
"""

def load_config():
    """Load environment variables from .env file"""
    load_dotenv()
    
    # Check for required environment variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        console.print("[bold red]ERROR:[/bold red] OpenAI API key not found.")
        console.print("Set your API key by:")
        console.print("  1. Creating a .env file with OPENAI_API_KEY=your_key_here")
        console.print("  2. Or setting environment variable: export OPENAI_API_KEY=your_key_here")
        console.print("\nRun [bold]cocode setup[/bold] to configure.")
        sys.exit(1)
    
    return {
        "api_key": api_key,
        "model": os.getenv("COCODE_MODEL", "gpt-4o"),
        "temperature": float(os.getenv("COCODE_TEMPERATURE", "0.2")),
        "work_dir": os.getenv("COCODE_WORK_DIR", os.getcwd()),
    }

@app.command()
def setup():
    """Configure CoCoDe with your API key and preferences"""
    console.print(Panel.fit(
        Text(BANNER, style="bright_blue"), 
        title="Welcome to CoCoDe", 
        subtitle="Setup", 
        border_style="blue"
    ))
    
    # Check if .env exists
    env_path = Path(".env")
    if env_path.exists():
        console.print("[yellow]Found existing .env file. Updating configuration...[/yellow]")
        with open(env_path, "r") as f:
            lines = f.readlines()
        
        # Parse existing values
        config = {}
        for line in lines:
            if "=" in line and not line.strip().startswith("#"):
                key, value = line.strip().split("=", 1)
                config[key] = value
    else:
        config = {}
    
    # Prompt for configuration
    api_key = typer.prompt("OpenAI API Key", default=config.get("OPENAI_API_KEY", ""), hide_input=True)
    model = typer.prompt("Default GPT model", default=config.get("COCODE_MODEL", "gpt-4o"))
    temperature = typer.prompt("Model temperature (0.0-1.0)", default=config.get("COCODE_TEMPERATURE", "0.2"))
    
    # Write configuration
    with open(env_path, "w") as f:
        f.write(f"OPENAI_API_KEY={api_key}\n")
        f.write(f"COCODE_MODEL={model}\n")
        f.write(f"COCODE_TEMPERATURE={temperature}\n")
    
    console.print("[green]Configuration saved successfully![/green]")
    console.print("Run [bold]cocode run[/bold] to start the assistant.")

@app.command()
def run(
    cwd: str = typer.Option(os.getcwd(), help="Directory to operate in"),
    model: str = typer.Option(None, help="GPT model to use (e.g., gpt-4o)"),
    temperature: float = typer.Option(None, help="Model temperature (0.0-1.0)"),
):
    """Run the CoCoDe assistant in the current directory"""
    # Load config
    config = load_config()
    if model:
        config["model"] = model
    if temperature is not None:
        config["temperature"] = temperature
    config["work_dir"] = cwd
    
    # Display banner
    console.print(Panel.fit(
        Text(BANNER, style="bright_blue"), 
        title="Welcome to CoCoDe", 
        subtitle=f"Using {config['model']}", 
        border_style="blue"
    ))
    
    # Initialize components
    console.print("[bold]Initializing components...[/bold]")
    
    # Initialize the model connector
    model_connector = ModelConnector(
        api_key=config["api_key"],
        model=config["model"],
        temperature=config["temperature"]
    )
    
    # Initialize the memory manager
    memory_manager = MemoryManager(work_dir=config["work_dir"])
    
    # Initialize Git agent
    git_agent = GitAgent(model_connector=model_connector)
    
    # Initialize file indexer
    file_indexer = FileIndexer(work_dir=config["work_dir"])
    
    # Initialize interpreter wrapper
    interpreter_wrapper = InterpreterWrapper(
        model_connector=model_connector,
        git_agent=git_agent,
        file_indexer=file_indexer,
        memory_manager=memory_manager
    )
    
    # Start the TUI
    console.print("[bold]Starting CoCoDe TUI...[/bold]")
    tui = TUIApp(
        interpreter=interpreter_wrapper,
        git_agent=git_agent,
        file_indexer=file_indexer,
        memory_manager=memory_manager,
        model_connector=model_connector,
        work_dir=config["work_dir"]
    )
    
    try:
        tui.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]CoCoDe terminated by user.[/yellow]")
        sys.exit(0)
    except Exception as e:
        console.print(f"\n[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    app()