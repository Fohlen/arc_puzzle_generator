"""
Visualization module for ARC puzzle generators.

This module provides functionality to visualize the output of ARC puzzle generators
using matplotlib for colorplots and tkinter for an interactive GUI.
"""
import argparse
import sys
import tkinter as tk
from tkinter import ttk
import importlib
from pathlib import Path
from typing import Optional, Type

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from arc_puzzle_generator.data_loader import load_puzzle, Puzzle
from arc_puzzle_generator.generators.generator import Generator


def plot_grid(grid: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Plot a grid using matplotlib.

    Args:
        grid: The grid to plot
        ax: Optional matplotlib axes to plot on

    Returns:
        The matplotlib axes object
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    colors = np.array([
        [0, 0, 0],
        [30, 147, 255],
        [249, 60, 49],
        [79, 204, 48],
        [255, 220, 0],
        [153, 153, 153],
        [229, 58, 163],
        [255, 133, 27],
        [135, 216, 241],
        [146, 18, 19]
    ]) / 255
    cmap = ListedColormap(colors)

    ax.imshow(grid, cmap=cmap, interpolation='nearest', vmin=0, vmax=9)

    # Add grid lines
    # ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=2)
    ax.set_xticks(np.arange(-0.5, grid.shape[1], 1))
    ax.set_yticks(np.arange(-0.5, grid.shape[0], 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Add cell values as text
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', color='w')

    return ax


class GeneratorVisualizer:
    """
    Tkinter-based GUI for visualizing generator steps.
    """

    def __init__(self, generator_class: Type[Generator], puzzle: Puzzle):
        """
        Initialize the visualizer.

        :arg generator_class: The generator class to use
        :arg puzzle: The puzzle containing train and test examples
        """
        self.generator_class = generator_class
        self.puzzle = puzzle
        self.generator = None
        self.iterator = None
        self.steps: list[np.ndarray] = []
        self.current_step = 0
        self.playing = False
        self.current_example_type = "train"
        self.current_example_index = 0

        # Create the main window
        self.root = tk.Tk()
        self.root.title("ARC Puzzle Generator Visualizer")
        self.root.geometry("800x800")

        # Create a parent frame to hold everything
        self.parent_frame = ttk.Frame(self.root)
        self.parent_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid layout for parent frame
        self.parent_frame.columnconfigure(0, weight=1)
        self.parent_frame.rowconfigure(0, weight=0)  # Selector row
        self.parent_frame.rowconfigure(1, weight=1)  # Main content
        self.parent_frame.rowconfigure(2, weight=0)  # Control row

        # Create the selector frame
        self.selector_frame = ttk.Frame(self.parent_frame, padding=10, relief="raised", borderwidth=1)
        self.selector_frame.grid(row=0, column=0, sticky="ew")

        # Create the puzzle selector dropdown
        self.example_var = tk.StringVar()
        self.example_options = []

        # Populate the dropdown options
        for i in range(len(self.puzzle.train)):
            self.example_options.append(f"Train {i + 1}")
        for i in range(len(self.puzzle.test)):
            self.example_options.append(f"Test {i + 1}")

        self.example_var.set(self.example_options[0])  # Set default value

        # Create the dropdown label
        self.example_label = ttk.Label(self.selector_frame, text="Select Puzzle:", font=('Arial', 12))
        self.example_label.pack(side=tk.LEFT, padx=10, pady=5)

        # Create the dropdown
        self.example_dropdown = ttk.Combobox(self.selector_frame, textvariable=self.example_var,
                                             values=self.example_options, state="readonly", font=('Arial', 12))
        self.example_dropdown.pack(side=tk.LEFT, padx=10, pady=5)
        self.example_dropdown.bind("<<ComboboxSelected>>", self.on_example_selected)

        # Create the main frame
        self.main_frame = ttk.Frame(self.parent_frame, padding=10)
        self.main_frame.grid(row=1, column=0, sticky="nsew")

        # Create the matplotlib figure
        self.figure = Figure(figsize=(8, 8))
        self.ax = self.figure.add_subplot(111)

        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create the control frame
        self.control_frame = ttk.Frame(self.parent_frame, padding=10, relief="raised", borderwidth=1)
        self.control_frame.grid(row=2, column=0, sticky="ew")

        # Create a style for the buttons
        style = ttk.Style()
        style.configure('Control.TButton', font=('Arial', 12))

        # Create the buttons with improved styling
        self.start_button = ttk.Button(self.control_frame, text="Start", command=self.go_to_start,
                                       style='Control.TButton')
        self.start_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.prev_button = ttk.Button(self.control_frame, text="Previous", command=self.previous_step,
                                      style='Control.TButton')
        self.prev_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_play,
                                      style='Control.TButton')
        self.play_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.next_button = ttk.Button(self.control_frame, text="Next", command=self.next_step, style='Control.TButton')
        self.next_button.pack(side=tk.LEFT, padx=10, pady=5)

        self.end_button = ttk.Button(self.control_frame, text="End", command=self.go_to_end, style='Control.TButton')
        self.end_button.pack(side=tk.LEFT, padx=10, pady=5)

        # Create the step label with improved styling
        self.step_label = ttk.Label(self.control_frame, text="Step: 0/0", font=('Arial', 12))
        self.step_label.pack(side=tk.RIGHT, padx=10, pady=5)

        # Initialize the generator
        self.initialize_generator()

    def on_example_selected(self, event):
        """Handle selection of a new example from the dropdown."""
        selected = self.example_var.get()

        # Parse the selection to determine type and index
        parts = selected.split()
        example_type = parts[0].lower()  # "train" or "test"
        example_index = int(parts[1]) - 1  # Convert to 0-based index

        # Update current selection
        self.current_example_type = example_type
        self.current_example_index = example_index

        # Reinitialize the generator with the new example
        self.initialize_generator()

    def initialize_generator(self):
        """Initialize the generator and prepare for visualization."""
        # Get the input grid for the selected example
        if self.current_example_type == "train":
            input_grid = self.puzzle.train[self.current_example_index].input
        else:  # test
            input_grid = self.puzzle.test[self.current_example_index].input

        # Create a new generator instance with the selected input
        self.generator = self.generator_class(input_grid)

        # Run the generator setup
        self.generator.setup()

        # Reset steps and current step
        self.steps = [self.generator.input_grid.copy()] + [step for step in self.generator]
        self.current_step = 0

        # Update the display
        self.update_display()

    def update_display(self):
        """Update the display with the current step."""
        if 0 <= self.current_step < len(self.steps):
            self.ax.clear()
            plot_grid(self.steps[self.current_step], self.ax)
            self.canvas.draw()
            self.step_label.config(text=f"Step: {self.current_step}/{len(self.steps) - 1}")

    def next_step(self):
        """Go to the next step."""
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            self.update_display()

    def previous_step(self):
        """Go to the previous step."""
        if self.current_step > 0:
            self.current_step -= 1
            self.update_display()

    def go_to_start(self):
        """Go to the first step."""
        self.current_step = 0
        self.update_display()

    def go_to_end(self):
        """Go to the last step by running the generator to completion."""
        self.current_step = len(self.steps) - 1
        self.update_display()

    def toggle_play(self):
        """Toggle automatic playback of steps."""
        self.playing = not self.playing
        if self.playing:
            self.play_button.config(text="Pause")
            self.play_next_step()
        else:
            self.play_button.config(text="Play")

    def play_next_step(self):
        """Play the next step and schedule the next one if still playing."""
        if self.playing:
            self.next_step()
            if self.current_step < len(self.steps) - 1 or self.iterator is not None:
                self.root.after(500, self.play_next_step)
            else:
                self.playing = False
                self.play_button.config(text="Play")

    def run(self):
        """Run the visualizer."""
        self.root.mainloop()


def get_generator_class(generator_name: str) -> Type[Generator]:
    """
    Get a generator class by name.

    :param generator_name: Name of the generator class
    :returns: The generator class
    """
    module = importlib.import_module("arc_puzzle_generator.generators")

    # Look for a class that matches the name
    if hasattr(module, generator_name):
        attr = getattr(module, generator_name)
        if isinstance(attr, type) and issubclass(attr, Generator) and attr != Generator:
            return attr

    raise AttributeError(f"No generator class found with name '{generator_name}'")


def visualize_generator(generator_name: str, puzzle_id: str, base_dir: str = "tests/data"):
    """
    Visualize a generator for a specific puzzle.

    :param generator_name: Name of the generator class
    :param puzzle_id: ID of the puzzle to visualize
    :param base_dir: Base directory for puzzles
    """
    # Load the puzzle
    puzzle_path = Path(base_dir) / f"{puzzle_id}.json"
    if not puzzle_path.exists():
        raise FileNotFoundError(f"Puzzle file not found: {puzzle_path}")

    puzzle = load_puzzle(puzzle_path)

    # Get the generator class
    generator_class = get_generator_class(generator_name)

    # Create the visualizer with the generator class and puzzle
    visualizer = GeneratorVisualizer(generator_class, puzzle)

    # Run the visualizer
    visualizer.run()


def main():
    """
    Main entry point for the visualization CLI.
    """
    parser = argparse.ArgumentParser(description="Visualize ARC puzzle generators")

    parser.add_argument(
        "generator",
        help="Name of the generator class to visualize"
    )

    parser.add_argument(
        "puzzle_id",
        help="ID of the puzzle to visualize"
    )

    parser.add_argument(
        "--base-dir",
        default="tests/data",
        help="Base directory for puzzles (default: tests/data)"
    )

    args = parser.parse_args()

    try:
        visualize_generator(
            args.generator,
            args.puzzle_id,
            args.base_dir,
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
