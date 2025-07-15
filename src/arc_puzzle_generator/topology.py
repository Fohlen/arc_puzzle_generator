from typing import Protocol

LabelSet = set[str]


class Topology(Protocol):
    """
    A topology is a callable that takes a label and a set of labels and returns a set of labels.
    It defines the relationships between agents in a system, such as which agents are connected or related to each other.
    """

    def __call__(self, label: str, labels: LabelSet) -> LabelSet:
        pass


def identity_topology(label: str, labels: LabelSet) -> LabelSet:
    """
    An identity topology that returns the label itself.

    :param label: The label to return.
    :param labels: The set of labels (not used).
    :return: A set containing the label itself.
    """
    return {label}


def all_topology(label: str, labels: LabelSet) -> LabelSet:
    """
    An all topology that returns all labels in the set.

    :param label: The label to return (not used).
    :param labels: The set of labels to return.
    :return: The set of all labels.
    """
    return {label} | labels


class FixedGroupTopology(Topology):
    """
    A fixed group topology that returns a fixed set of labels.
    This is useful for defining a specific group of agents that are always considered together.
    """

    def __init__(self, group: LabelSet):
        self.group = group

    def __call__(self, label: str, labels: LabelSet) -> LabelSet:
        return self.group
