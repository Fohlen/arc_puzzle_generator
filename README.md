arc_puzzle_generator
--------------------

This repository contains a general purpose agentic model framework suitable for
the [Abstraction and Reasoning Corpus](https://github.com/arcprize/ARC-AGI-2) puzzles.

[![CI](https://github.com/Fohlen/arc_puzzle_generator/actions/workflows/ci.yml/badge.svg)](https://github.com/Fohlen/arc_puzzle_generator/actions/workflows/ci.yml)

## Installation and use

One can install the package using pip:

```shell
pip install -e .
```

This will give you access to individual puzzles and the functions that power them, for instance:

```python
from pathlib import Path
from arc_puzzle_generator.utils.data_loader import load_puzzle
from arc_puzzle_generator.puzzles import puzzle_two

file_path = Path("3e6067c3.json")
puzzle = load_puzzle(file_path)

playground = puzzle_two(puzzle.train[0].input)
*_, output_grid = playground
```

The `playground` creates an `Iterable[np.ndarray]` which will yield one output grid for every agentic step taken on the
input grid.
The last state corresponds to the final solution.

## Visualization

This package comes with a simple visualization tool that can be used to visualize the output of the models.
You can install it by using `pip` and a package manager of your choice:

```shell
brew install python3-tk
pip install -e ".[vis]"
```

Once dependencies for `tkinter` are satisfied, you can run the visualization tool by running:

```shell
python3 -m arc_puzzle_generator.visualization puzzle_two 3e6067c3
```

The `--help` flag will give you more information about the tool.

## Development

To set up the development environment:

```shell
pip install -e ".[dev]"
```

To run tests:

```shell
python -m unittest discover tests
```

To run type checking:

```shell
python -m mypy src tests
```

These checks are automatically run on each commit via GitHub Actions.
