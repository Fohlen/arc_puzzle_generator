import json
from pathlib import Path
from typing import TypedDict, cast
from dataclasses import dataclass

import numpy as np


class RawPair(TypedDict):
    input: list[list[int]]
    output: list[list[int]]


class RawPuzzle(TypedDict):
    train: list[RawPair]
    test: list[RawPair]


@dataclass(frozen=True)
class Pair:
    input: np.ndarray
    output: np.ndarray


@dataclass(frozen=True)
class Puzzle:
    train: list[Pair]
    test: list[Pair]


def load_puzzle(puzzle_file: Path) -> Puzzle:
    """
    Loads a puzzle from a JSON file. See [Task file format](https://github.com/arcprize/ARC-AGI-2/tree/main?tab=readme-ov-file) for more information.
    :param puzzle_file: The path to the JSON file.
    :return: A puzzle dictionary.
    """

    with puzzle_file.open() as f:
        raw_puzzle = cast(RawPuzzle, json.load(f))
        return Puzzle(
            train=[Pair(
                input=np.array(pair["input"]),
                output=np.array(pair["output"])
            ) for pair in raw_puzzle["train"]],
            test=[Pair(
                input=np.array(pair["input"]),
                output=np.array(pair["output"])
            ) for pair in raw_puzzle["test"]]
        )
