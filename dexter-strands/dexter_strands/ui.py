"""Rich terminal UI for Dexter agent.

This module provides a comprehensive UI layer using the Rich library for
beautiful terminal output with colors, boxes, tables, and animations.
"""

from contextlib import contextmanager
from typing import Iterator, List, Dict, Any, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
from rich.text import Text
from rich import box


class RichUI:
    """Rich terminal UI for Dexter agent.
    
    Provides methods for displaying formatted output including:
    - User queries
    - Task lists
    - Progress indicators
    - Streaming answers
    - Financial data tables
    - Error messages
    """
    
    def __init__(self, width: Optional[int] = None):
        """Initialize Rich console with configuration.
        
        Args:
            width: Optional console width. If None, uses terminal width.
        """
        self.console = Console(
            width=width,
            color_system="auto",  # Auto-detect terminal color support
            force_terminal=True,   # Ensure Rich formatting is enabled
            legacy_windows=False   # Use modern Windows terminal support
        )
    
    def print_user_query(self, query: str) -> None:
        """Display user's query in bold light blue.
        
        Args:
            query: The user's query text to display
        """
        self.console.print(f"[bold light_blue]{query}[/]")
    
    def print_task_list(self, tasks: List[Dict[str, Any]]) -> None:
        """Display task list in condensed format matching legacy style.
        
        Args:
            tasks: List of task dictionaries with 'description' and 'status' keys
                   Expected keys: 'description' (str), 'status' (str)
                   Status values: 'pending', 'active', 'complete', 'failed'
        """
        # Map status to symbol and color
        status_map = {
            'pending': ('+', 'dim'),
            'active': ('▶', 'cyan'),
            'complete': ('✓', 'green'),
            'failed': ('✗', 'red')
        }
        
        # Print header with left-aligned title (legacy style)
        self.console.print(f"\n[bold blue]╭─ Planned Tasks[/]")
        
        # Print each task with minimal formatting
        for task in tasks:
            description = task.get('description', '')
            status = task.get('status', 'pending')
            symbol, color = status_map.get(status, ('+', 'dim'))
            self.console.print(f"[blue]│[/] [{color}]{symbol}[/] {description}")
        
        # Print footer with simple border
        self.console.print(f"[blue]╰{'─' * 50}[/]\n")
    
    @contextmanager
    def progress(self, message: str, success_message: str = ""):
        """Context manager for progress spinner.
        
        Args:
            message: Message to display during progress
            success_message: Optional message to display on success
            
        Yields:
            None
            
        Example:
            with ui.progress("Processing"):
                # Do work
                pass
                
        Raises:
            Exception: Re-raises any exception after displaying error
        """
        # Start spinner with cyan color
        with self.console.status(f"[cyan]{message}...[/]", spinner="dots") as status:
            try:
                yield
                # Success - display checkmark
                if success_message:
                    self.console.print(f"[green]✓[/] {success_message}")
                else:
                    self.console.print(f"[green]✓[/] {message} complete")
            except Exception as e:
                # Error - display X and re-raise
                self.console.print(f"[red]✗[/] {message} failed: {str(e)}")
                raise
    
    @contextmanager
    def coordinated_progress(self, message: str):
        """Coordinated progress context that manages a single Live context.
        
        This context manager prevents multiple spinners from conflicting during
        multi-agent handoffs by using Rich's Live context for in-place updates.
        All nested operations share this Live context, ensuring updates happen
        in place without printing new lines.
        
        Args:
            message: Message to display during progress
            
        Yields:
            Live: Rich Live context that can be updated by nested operations
            
        Example:
            with ui.coordinated_progress("Processing query"):
                # All nested operations share this Live context
                # Updates happen in place
                pass
                
        Raises:
            Exception: Re-raises any exception after ensuring proper cleanup
            
        Requirements:
            - 4.1: Updates spinner animation in place without printing new lines
            - 4.2: Replaces spinner with completion indicator on same line
            - 4.5: Ensures proper context management to prevent spinner verbosity
        """
        # Create spinner with cyan color and message
        spinner = Spinner("dots", text=f"[cyan]{message}...[/]", style="cyan")
        
        # Create Live context for in-place updates
        # refresh_per_second=10 provides smooth animation (Requirement 10.1)
        with Live(
            spinner,
            console=self.console,
            refresh_per_second=10,
            transient=False  # Keep final output visible
        ) as live:
            try:
                # Yield the Live context so nested operations can update it
                yield live
                
                # Success - update with checkmark on same line (Requirement 4.2)
                success_text = Text()
                success_text.append("✓", style="green")
                success_text.append(f" {message} complete")
                live.update(success_text)
                
                # Stop the live context to finalize the output
                live.stop()
                
            except Exception as e:
                # Error - update with X symbol on same line (Requirement 4.2)
                error_text = Text()
                error_text.append("✗", style="red")
                error_text.append(f" {message} failed: {str(e)}")
                live.update(error_text)
                
                # Stop the live context to finalize the output
                live.stop()
                
                # Re-raise the exception
                raise
    
    def stream_answer(self, text_chunks: Iterator[str]) -> str:
        """Stream answer text in formatted box, return accumulated text.
        
        This method displays text chunks in real-time within a double-border box,
        handling word wrapping and newline preservation.
        
        Args:
            text_chunks: Iterator of text chunks to stream
            
        Returns:
            Complete accumulated text
            
        Raises:
            Exception: Re-raises any exception after ensuring box is properly closed
        """
        # Pre-calculate box dimensions (Performance optimization: 11.1)
        width = 80
        content_width = width - 4  # Account for borders and padding
        
        # Pre-calculate static strings to avoid repeated string operations
        top_border = f"\n[bold blue]╔{'═' * (width - 2)}╗[/]"
        separator = f"[blue]╠{'═' * (width - 2)}╣[/]"
        bottom_border = f"[bold blue]╚{'═' * (width - 2)}╝[/]\n"
        left_border = "[blue]║[/]"
        right_border = " [blue]║[/]"
        
        # Pre-calculate title line
        title = "ANSWER"
        padding = (width - len(title) - 2) // 2
        title_line = f"[bold blue]║{' ' * padding}{title}{' ' * (width - len(title) - padding - 2)}║[/]"
        
        # Initialize box borders (top, title, separator)
        self.console.print(top_border)
        self.console.print(title_line)
        self.console.print(separator)
        
        # Start content area - print first line border
        self.console.print(left_border, end="")
        
        # Prepare for character-by-character processing
        accumulated_text = ""
        current_line = ""
        
        # Optimize flush frequency (Performance optimization: 11.1)
        # Track characters since last flush to minimize buffering
        chars_since_flush = 0
        flush_threshold = 50  # Flush every 50 characters for responsive display
        
        try:
            # Process chunks from iterator
            for chunk in text_chunks:
                # Accumulate text character-by-character
                accumulated_text += chunk
                
                # Process each character in the chunk
                for char in chunk:
                    chars_since_flush += 1
                    
                    # Handle newline characters
                    if char == '\n':
                        # Complete current line with padding
                        padding_needed = max(0, content_width - len(current_line))
                        self.console.print(
                            f" {current_line}{' ' * padding_needed}{right_border}"
                        )
                        # Start new line with border
                        self.console.print(left_border, end="")
                        current_line = ""
                        chars_since_flush = 0  # Reset flush counter after newline
                    else:
                        current_line += char
                        
                        # Detect line width limits and implement word wrapping logic
                        if len(current_line) >= content_width:
                            # Find word boundaries for wrapping
                            last_space = current_line.rfind(' ', 0, content_width)
                            
                            if last_space > 0:
                                # Wrap at word boundary
                                line_to_print = current_line[:last_space]
                                padding_needed = content_width - len(line_to_print)
                                self.console.print(
                                    f" {line_to_print}{' ' * padding_needed}{right_border}"
                                )
                                self.console.print(left_border, end="")
                                current_line = current_line[last_space + 1:]
                            else:
                                # Handle edge case of words longer than line width
                                # Wrap at character boundary
                                line_to_print = current_line[:content_width]
                                padding_needed = content_width - len(line_to_print)
                                self.console.print(
                                    f" {line_to_print}{' ' * padding_needed}{right_border}"
                                )
                                self.console.print(left_border, end="")
                                current_line = current_line[content_width:]
                            
                            chars_since_flush = 0  # Reset flush counter after line wrap
                    
                    # Optimize flush frequency: flush every N characters for responsive display
                    # This minimizes buffering while avoiding excessive flush calls
                    if chars_since_flush >= flush_threshold:
                        self.console.file.flush()
                        chars_since_flush = 0
            
            # Print any remaining content on the current line
            if current_line:
                padding_needed = max(0, content_width - len(current_line))
                self.console.print(f" {current_line}{' ' * padding_needed}{right_border}")
            else:
                # Empty line at end
                self.console.print(f"{' ' * (width - 2)}{right_border}")
                
        except Exception as e:
            # Handle streaming errors gracefully
            # Ensure box is always properly closed
            if current_line:
                padding_needed = max(0, content_width - len(current_line))
                self.console.print(f" {current_line}{' ' * padding_needed}{right_border}")
            else:
                self.console.print(f"{' ' * (width - 2)}{right_border}")
            
            # Add error line
            error_msg = f"Error during streaming: {str(e)}"
            padding_needed = max(0, content_width - len(error_msg))
            self.console.print(f"{left_border} [red]{error_msg}[/]{' ' * padding_needed}{right_border}")
            
            # Close box before re-raising
            self.console.print(f"{left_border}{' ' * (width - 2)}{right_border}")
            self.console.print(bottom_border)
            raise
        
        # Close the box properly
        self.console.print(f"{left_border}{' ' * (width - 2)}{right_border}")
        self.console.print(bottom_border)
        
        # Return accumulated text
        return accumulated_text
    
    def format_table(
        self, 
        data: List[Dict], 
        title: str = "", 
        max_rows: Optional[int] = 100,
        max_col_width: Optional[int] = 50
    ) -> Table:
        """Format financial data as Rich table with performance optimizations.
        
        Args:
            data: List of dictionaries containing table data
            title: Optional table title
            max_rows: Maximum number of rows to display (default: 100, None for unlimited)
            max_col_width: Maximum column width (default: 50, None for unlimited)
            
        Returns:
            Rich Table object
        """
        # Performance optimization: 11.3 - Row limit for display
        # Limit rows to prevent performance issues with large datasets
        display_data = data[:max_rows] if max_rows and len(data) > max_rows else data
        truncated = max_rows and len(data) > max_rows
        
        # Create Rich Table with columns
        table = Table(
            title=title if title else None,
            box=box.ROUNDED,
            border_style="blue",
            header_style="bold cyan",
            show_lines=False  # Performance: disable row lines for faster rendering
        )
        
        # If no data, return empty table
        if not display_data:
            return table
        
        # Extract column names from first row
        columns = list(display_data[0].keys())
        
        # Performance optimization: 11.3 - Column width optimization
        # Pre-calculate column types and widths to avoid repeated checks
        column_info = {}
        for col in columns:
            # Check if column contains numeric data (sample first 10 rows for performance)
            sample_rows = display_data[:min(10, len(display_data))]
            is_numeric = all(
                isinstance(row.get(col), (int, float)) or 
                (isinstance(row.get(col), str) and 
                 row.get(col).replace(',', '').replace('.', '').replace('-', '').replace('$', '').strip().replace(' ', '').isdigit())
                for row in sample_rows if col in row and row.get(col)
            )
            
            # Calculate optimal column width (sample first 20 rows)
            sample_for_width = display_data[:min(20, len(display_data))]
            max_content_width = max(
                len(str(row.get(col, ""))) 
                for row in sample_for_width
            ) if sample_for_width else 10
            
            # Apply max_col_width limit if specified
            optimal_width = min(max_content_width, max_col_width) if max_col_width else max_content_width
            
            column_info[col] = {
                'is_numeric': is_numeric,
                'width': optimal_width
            }
        
        # Add columns with optimized settings
        for col in columns:
            info = column_info[col]
            justify = "right" if info['is_numeric'] else "left"
            
            # Set max_width to prevent extremely wide columns
            table.add_column(
                col, 
                justify=justify,
                max_width=info['width'],
                overflow="fold"  # Fold long content instead of truncating
            )
        
        # Performance optimization: 11.3 - Lazy evaluation for large tables
        # Add rows efficiently without unnecessary string operations
        for row in display_data:
            # Convert row values to strings, truncate if needed
            row_values = []
            for col in columns:
                value = row.get(col, "")
                str_value = str(value)
                
                # Truncate individual cell values if they exceed column width
                if max_col_width and len(str_value) > max_col_width:
                    str_value = self._truncate_output(str_value, max_col_width)
                
                row_values.append(str_value)
            
            table.add_row(*row_values)
        
        # Add footer note if data was truncated
        if truncated:
            table.caption = f"[dim]Showing {len(display_data)} of {len(data)} rows[/]"
        
        return table
    
    def format_number(self, value: float, is_currency: bool = False) -> str:
        """Format number with commas and color coding.
        
        Args:
            value: Numerical value to format
            is_currency: Whether to format as currency
            
        Returns:
            Formatted string with color markup
        """
        # Add comma separators for thousands
        # Format with commas - handle both integers and floats
        if isinstance(value, float) and value != int(value):
            # Float with decimal places
            formatted = f"{value:,.2f}"
        else:
            # Integer or whole number
            formatted = f"{int(value):,}"
        
        # Handle currency formatting
        if is_currency:
            formatted = f"${formatted}"
        
        # Apply green color for positive numbers
        if value > 0:
            return f"[green]{formatted}[/]"
        # Apply red color for negative numbers
        elif value < 0:
            return f"[red]{formatted}[/]"
        else:
            # Zero - no color
            return formatted
    
    def print_tool_params(self, tool_name: str, params: Dict[str, Any]) -> None:
        """Display tool parameters with → symbol in magenta.
        
        Args:
            tool_name: Name of the tool being called
            params: Dictionary of tool parameters
        """
        # Format parameters as key=value pairs with dim styling
        param_strs = [f"{key}={value}" for key, value in params.items()]
        params_text = ", ".join(param_strs) if param_strs else "no parameters"
        
        # Display with → symbol in magenta and dim parameter values
        self.console.print(f"[magenta]→[/] {tool_name}([dim]{params_text}[/])")
    
    def print_tool_run(self, tool_name: str, result: str, max_length: int = 200) -> None:
        """Display tool result with ⚡ symbol in yellow.
        
        Args:
            tool_name: Name of the tool that was executed
            result: Tool execution result (will be truncated if too long)
            max_length: Maximum length before truncation (default: 200)
        """
        # Truncate long results with ellipsis (Performance optimization: 11.2)
        truncated_result = self._truncate_output(result, max_length)
        
        # Display with ⚡ symbol in yellow
        self.console.print(f"[yellow]⚡[/] {tool_name}: {truncated_result}")
    
    def _truncate_output(self, text: str, max_length: int) -> str:
        """Truncate output text while maintaining readability.
        
        This helper method implements smart truncation that:
        - Adds ellipsis for truncated content
        - Maintains readability by not cutting mid-word when possible
        - Preserves formatting markers
        
        Args:
            text: Text to truncate
            max_length: Maximum length before truncation
            
        Returns:
            Truncated text with ellipsis if needed
        """
        # Performance optimization: 11.2 - Implement max length limits
        if len(text) <= max_length:
            return text
        
        # Truncate and add ellipsis
        # Try to find a space near the end to avoid cutting mid-word
        truncate_point = max_length - 3  # Reserve space for "..."
        
        # Look for a space in the last 20 characters to break cleanly
        last_space = text.rfind(' ', max(0, truncate_point - 20), truncate_point)
        
        if last_space > 0 and last_space > truncate_point - 20:
            # Found a good break point
            return text[:last_space] + "..."
        else:
            # No good break point, just truncate at max length
            return text[:truncate_point] + "..."
    
    def print_task_start(self, task_description: str) -> None:
        """Display task start indicator with ▶ symbol.
        
        Args:
            task_description: Description of the task starting
        """
        # Display with ▶ symbol in cyan (active operation color)
        self.console.print(f"[cyan]▶[/] {task_description}")
    
    def print_task_done(self, task_description: str) -> None:
        """Display task completion indicator with ✓ symbol.
        
        Args:
            task_description: Description of the completed task
        """
        # Display with ✓ symbol in green (success color)
        self.console.print(f"[green]✓[/] {task_description}")
    
    def print_task_error(self, task_description: str, error_message: str, max_length: int = 100) -> None:
        """Display task error indicator with ✗ symbol.
        
        This method displays a task failure with proper visual hierarchy,
        showing the task description and a brief error message.
        
        Args:
            task_description: Description of the failed task
            error_message: Error message explaining the failure
            max_length: Maximum length for error message before truncation (default: 100)
            
        Requirements:
            - 3.4: Display ✗ symbol with error message when task fails
            - 8.1: Display error with distinctive symbol (✗ in red)
            - 8.3: Provide context about which task failed
            - 8.4: Maintain visual hierarchy and formatting consistency
        """
        # Truncate long error messages for readability
        # Requirements: 8.4
        truncated_error = self._truncate_output(error_message, max_length)
        
        # Display with ✗ symbol in red (error color)
        # Maintain consistent indentation (0 spaces for top-level)
        # Requirements: 3.4, 8.1, 8.3, 8.4
        self.console.print(f"[red]✗[/] {task_description}")
        
        # Display error message with indentation (2 spaces for sub-item)
        # Requirements: 8.3, 8.4
        self.console.print(f"  [dim red]{truncated_error}[/]")
    
    def print_error(self, message: str, suggestion: str = "", max_length: int = 500) -> None:
        """Display formatted error message with optional suggestion.
        
        Args:
            message: Error message to display
            suggestion: Optional suggestion for resolving the error
            max_length: Maximum length for error message before truncation (default: 500)
        """
        # Truncate long error messages (Performance optimization: 11.2)
        truncated_message = self._truncate_output(message, max_length)
        
        # Build content with error message and optional suggestion
        if suggestion:
            content = f"[red]{truncated_message}[/]\n\n[dim]{suggestion}[/]"
        else:
            content = f"[red]{truncated_message}[/]"
        
        # Display in Panel with red border and error symbol
        panel = Panel(
            content,
            title="[bold red]✗ Error[/]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def print_agent_transition(self, from_agent: str, to_agent: str, reason: str = "") -> None:
        """Display agent transition indicator.
        
        This method displays when control passes between agents in the multi-agent
        workflow, helping users understand the orchestration process.
        
        Args:
            from_agent: Name of the agent handing off control
            to_agent: Name of the agent receiving control
            reason: Optional reason for the transition (e.g., "handoff", "insufficient data")
            
        Requirements:
            - 7.1: Display transition when Planner hands off to Executor
            - 7.2: Display transition when Executor hands off to Synthesizer
            - 7.3: Display transition when Synthesizer hands off to Planner
            - 7.4: Use consistent formatting and symbols for all transitions
        """
        # Build transition message with consistent formatting
        # Use → symbol and cyan color for transitions
        # Requirements: 7.4
        if reason:
            transition_text = f"[cyan]→ GRAPH TRANSITION:[/] {from_agent} → {to_agent} [dim]({reason})[/]"
        else:
            transition_text = f"[cyan]→ GRAPH TRANSITION:[/] {from_agent} → {to_agent}"
        
        # Display with consistent indentation (0 spaces for top-level)
        # Requirements: 7.4
        self.console.print(transition_text)
