arc_puzzle_generator
--------------------

This repository contains a puzzle generator for
the [Abstraction and Reasoning Corpus](https://github.com/arcprize/ARC-AGI-2).

## Installation and use

One can install the package using pip:

```shell
pip install -e .
```

This will give you access to individual generators and the functions that power them, for instance:

```python
from pathlib import Path
from arc_puzzle_generator.data_loader import load_puzzle
from arc_puzzle_generator.generators import PuzzleTwoGenerator

file_path = Path("3e6067c3.json")
puzzle = load_puzzle(file_path)

generator = PuzzleTwoGenerator(puzzle.train[0].input)
generator.setup()
*_, output_grid = generator
```

The generator creates an `Iterable[np.ndarray]` which will yield one output grid for every agentic step taken on the
input grid.
The last state corresponds to the final solution.

## Visualization

This package comes with a simple visualization tool that can be used to visualize the output of the generators.
You can install it by using `pip` and a package manager of your choice:

```shell
brew install python3-tk
pip install -e ".[vis]"
```

Once dependencies for `tkinter` are satisfied, you can run the visualization tool by running:

```shell
python3 -m arc_puzzle_generator.visualization PuzzleFourGenerator 142ca369
```

The `--help` flag will give you more information about the tool.
