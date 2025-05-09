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

from src.arc_puzzle_generator.data_loader import load_puzzle, Puzzle
from src.arc_puzzle_generator.generators.generator import Generator


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

    # Fixed color mapping as per requirements
    color_mapping = {
        0: (0, 0, 0),
        1: (30, 147, 255),
        2: (249, 60, 49),
        3: (79, 204, 48),
        4: (255, 220, 0),
        5: (153, 153, 153),
        6: (229, 58, 163),
        7: (255, 133, 27),
        8: (135, 216, 241),
        9: (146, 18, 19)
    }

    # Create a custom colormap based on the mapping
    colors = list(color_mapping.values())
    cmap = ListedColormap(colors)

    ax.imshow(grid, cmap=cmap, interpolation='nearest')

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

    def __init__(self, generator: Generator):
        """
        Initialize the visualizer.

        :arg generator: The generator to visualize
        """
        self.generator = generator
        self.iterator = None
        self.steps = []
        self.current_step = 0
        self.playing = False

        # Create the main window
        self.root = tk.Tk()
        self.root.title("ARC Puzzle Generator Visualizer")
        self.root.geometry("800x800")

        # Create a parent frame to hold everything
        self.parent_frame = ttk.Frame(self.root)
        self.parent_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid layout for parent frame
        self.parent_frame.columnconfigure(0, weight=1)
        self.parent_frame.rowconfigure(0, weight=1)
        self.parent_frame.rowconfigure(1, weight=0)  # Control row doesn't need to expand

        # Create the main frame
        self.main_frame = ttk.Frame(self.parent_frame, padding=10)
        self.main_frame.grid(row=0, column=0, sticky="nsew")

        # Create the matplotlib figure
        self.figure = Figure(figsize=(8, 8))
        self.ax = self.figure.add_subplot(111)

        # Create the canvas
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Create the control frame
        self.control_frame = ttk.Frame(self.parent_frame, padding=10, relief="raised", borderwidth=1)
        self.control_frame.grid(row=1, column=0, sticky="ew")

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

    def initialize_generator(self):
        """Initialize the generator and prepare for visualization."""
        # Run the generator setup
        self.generator.setup()

        self.steps = [self.generator.input_grid.copy()] + [step for step in self.generator]

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

    # Create the generator with the first training input
    generator = generator_class(puzzle.train[0].input)

    # Create the visualizer
    visualizer = GeneratorVisualizer(generator)

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
