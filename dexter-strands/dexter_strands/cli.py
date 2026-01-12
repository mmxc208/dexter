"""Command-line interface for Dexter Strands agent.

This module provides a simple CLI for interacting with the Dexter financial
research agent. It handles user input, displays progress during processing,
and shows agent responses.

Requirements:
    - 5.1: Display welcome message with usage instructions
    - 5.2: Accept text input from command line
    - 5.3: Display simple progress indicator
    - 5.4: Print agent response to terminal
    - 5.5: Accept "exit" or "quit" commands to terminate session
"""

import sys
import threading
import time
from typing import Optional

from dotenv import load_dotenv
from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory

from .agent import MultiAgentDexter
from .config import Config, ConfigurationError
from .context import ContextManager
from .logger import setup_logger, log_info, log_error
from .model import create_bedrock_model
from .ui import RichUI


def display_welcome(ui: RichUI) -> None:
    """Display welcome message and usage instructions using RichUI.
    
    Shows a friendly welcome message with basic usage instructions
    and available commands formatted with Rich Panel.
    
    Args:
        ui: RichUI instance for formatted output
    
    Requirements:
        - 1.1: Display welcome message with usage instructions
        - 5.1: Display welcome message with usage instructions
    """
    from rich.panel import Panel
    from rich import box
    
    welcome_content = """[bold cyan]Welcome! I'm Dexter, your AI-powered financial research assistant.[/]

I can help you analyze companies, understand financial data, and answer
questions about markets and investments.

[bold]USAGE:[/]
  • Type your question and press Enter
  • I'll process your query and provide detailed answers
  • Type [bold]'exit'[/] or [bold]'quit'[/] to end the session
  • Press [bold]Ctrl+C[/] or [bold]Ctrl+D[/] to interrupt at any time
  • Use [bold]UP/DOWN[/] arrows to navigate command history

[bold]EXAMPLES:[/]
  • "What is Apple's revenue for the last quarter?"
  • "Compare Microsoft and Google's profit margins"
  • "Show me Tesla's balance sheet"

[dim]Let's get started![/]
"""
    
    panel = Panel(
        welcome_content,
        title="[bold blue]DEXTER - Financial Research Assistant (Strands Edition)[/]",
        border_style="blue",
        box=box.DOUBLE,
        padding=(1, 2)
    )
    
    ui.console.print(panel)


def get_user_input(prompt: str = ">> ") -> str:
    """Get query from user with simple prompt.
    
    Args:
        prompt: Prompt string to display (default: ">> ")
        
    Returns:
        User input string (stripped of whitespace)
        
    Requirements:
        - 5.2: Accept text input from command line
    """
    try:
        user_input = input(prompt)
        return user_input.strip()
    except EOFError:
        # Handle Ctrl+D
        return "exit"


class ProgressIndicator:
    """Simple progress indicator with spinner animation.
    
    Displays a rotating spinner while the agent is processing a query.
    The spinner runs in a separate thread and can be stopped when
    processing is complete.
    
    Requirements:
        - 5.3: Display simple progress indicator
    """
    
    def __init__(self, message: str = "Processing"):
        """Initialize progress indicator.
        
        Args:
            message: Message to display alongside spinner
        """
        self.message = message
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.current_idx = 0
    
    def start(self) -> None:
        """Start the progress indicator animation."""
        self.running = True
        self.thread = threading.Thread(target=self._animate, daemon=True)
        self.thread.start()
    
    def stop(self) -> None:
        """Stop the progress indicator animation."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        # Clear the line
        sys.stdout.write("\r" + " " * (len(self.message) + 10) + "\r")
        sys.stdout.flush()
    
    def _animate(self) -> None:
        """Animation loop (runs in separate thread)."""
        while self.running:
            spinner = self.spinner_chars[self.current_idx]
            sys.stdout.write(f"\r{spinner} {self.message}...")
            sys.stdout.flush()
            self.current_idx = (self.current_idx + 1) % len(self.spinner_chars)
            time.sleep(0.1)


def display_progress(message: str = "Processing") -> ProgressIndicator:
    """Show simple progress indicator during processing.
    
    Creates and starts a progress indicator that displays a spinner
    animation. The caller should call stop() on the returned indicator
    when processing is complete.
    
    Args:
        message: Message to display alongside spinner
        
    Returns:
        ProgressIndicator instance (already started)
        
    Requirements:
        - 5.3: Display simple progress indicator
        
    Example:
        >>> progress = display_progress("Thinking")
        >>> # ... do work ...
        >>> progress.stop()
    """
    indicator = ProgressIndicator(message)
    indicator.start()
    return indicator


def display_response(response: str) -> None:
    """Print agent response to terminal.
    
    Displays the agent's response in a formatted box for better readability.
    
    Args:
        response: Agent response text to display
        
    Requirements:
        - 5.4: Print agent response to terminal
    """
    # Simple box formatting
    print("\n" + "─" * 70)
    print(response)
    print("─" * 70 + "\n")


def main() -> None:
    """Main CLI loop.
    
    This is the entry point for the CLI application. It:
    1. Loads environment configuration
    2. Initializes the agent
    3. Displays welcome message
    4. Enters interactive loop to process user queries
    5. Handles exit commands and errors
    
    Requirements:
        - 5.1: Display welcome message
        - 5.2: Accept text input
        - 5.3: Display progress indicator
        - 5.4: Print agent responses
        - 5.5: Handle exit/quit commands
    """
    # Load environment variables from .env file
    load_dotenv()
    
    try:
        # Load configuration (includes Langfuse tracing settings)
        # Requirements: 7.1, 7.4
        config = Config.from_env()
        
        # Setup logging
        setup_logger(
            log_level=config.log_level,
            log_file=config.log_file
        )
        
        log_info("Starting Dexter Strands CLI")
        
        # Log tracing configuration status
        # Requirements: 7.1, 7.4, 10.3
        if config.langfuse_enabled:
            if config.langfuse_public_key and config.langfuse_secret_key:
                log_info(f"Tracing enabled: {config.langfuse_host} (environment: {config.trace_environment})")
            else:
                log_info("Tracing enabled in config but credentials missing - will be disabled")
        else:
            log_info("Tracing disabled via configuration")
        
        # Create Bedrock model
        model = create_bedrock_model(config)
        
        # Import all financial tools from registry
        from .tools import ALL_TOOLS
        
        tools = ALL_TOOLS
        
        # Initialize RichUI for agent
        ui = RichUI()
        
        # Initialize ContextManager for context offloading
        context_manager = ContextManager(
            context_dir=config.context_dir
        )
        
        # Create multi-agent system with UI and context management
        # Tracing setup is called automatically in MultiAgentDexter.__init__
        # Requirements: 7.1, 7.3, 7.4, 9.1, 9.2, 10.3
        agent = MultiAgentDexter(
            model=model,
            tools=tools,
            config=config,
            context_manager=context_manager,
            ui=ui
        )
        
        log_info("Multi-agent system initialized successfully")
        
        # Log final tracing status after agent initialization
        # Requirements: 7.4, 10.3
        if agent.tracing_enabled:
            log_info("Tracing active - traces will be sent to Langfuse")
        else:
            log_info("Tracing inactive - agent will run without observability")
        
    except ConfigurationError as e:
        print(f"\n❌ Configuration Error:\n{e}\n")
        print("Please check your .env file and ensure all required variables are set.")
        print("See .env.example for reference.\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Initialization Error: {e}\n")
        sys.exit(1)
    
    # Initialize prompt_toolkit session with history
    # Requirements: 1.2, 7.1, 7.2
    session = PromptSession(history=InMemoryHistory())
    
    # Display welcome message
    # Requirement: 1.1, 5.1
    display_welcome(ui)
    
    # Main interaction loop
    while True:
        try:
            # Get user input with prompt_toolkit session
            # Requirements: 1.2, 1.5, 7.2
            query = session.prompt(">> ").strip()
            
            # Check for exit commands
            # Requirement: 1.5, 5.5
            if query.lower() in ["exit", "quit", ""]:
                if query.lower() in ["exit", "quit"]:
                    ui.console.print("\n[cyan]👋 Goodbye! Thanks for using Dexter.[/]\n")
                    log_info("User exited CLI")
                break
            
            # Display user query with Rich formatting
            ui.print_user_query(query)
            
            # Show progress indicator using RichUI
            # Requirements: 7.3, 7.4
            with ui.progress("Processing your query"):
                try:
                    # Process query with agent
                    response = agent.process_query(query)
                    
                except Exception as e:
                    # Display error message with RichUI
                    ui.print_error(
                        str(e),
                        "Please check your query and try again, or contact support if the issue persists."
                    )
                    log_error(e)
                    continue
            
            # Display response with Rich formatting
            # Requirement: 7.4
            from rich.panel import Panel
            from rich import box
            
            response_panel = Panel(
                response,
                title="[bold blue]Response[/]",
                border_style="blue",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            ui.console.print(response_panel)
                
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            # Requirement: 1.5
            ui.console.print("\n\n[cyan]👋 Interrupted. Goodbye![/]\n")
            log_info("User interrupted CLI with Ctrl+C")
            break
        except EOFError:
            # Handle Ctrl+D gracefully
            # Requirement: 1.5
            ui.console.print("\n\n[cyan]👋 Goodbye![/]\n")
            log_info("User exited CLI with Ctrl+D")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}\n")
            log_error(e)
            # Continue the loop instead of crashing


if __name__ == "__main__":
    main()
