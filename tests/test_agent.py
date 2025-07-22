import unittest
from arc_puzzle_generator.agent import Agent
from arc_puzzle_generator.geometry import PointSet
from arc_puzzle_generator.rule import RuleNode
from arc_puzzle_generator.state import AgentState


def dummy_rule(states, colors, collision, collision_mapping):
    """
    Always moves the agent to the right, consumes the next color from the iterator, and decrements the charge.
    """
    last = states[-1]
    new_pos = PointSet([(p[0], p[1] + 1) for p in last.position])
    new_color = next(colors)
    return AgentState(new_pos, "right", new_color, last.charge - 1), colors


def collision_rule(states, colors, collision, collision_mapping):
    """
    Rule checks collision and consumes color if there is a collision.
    """

    last = states[-1]
    if collision:
        new_color = next(colors)
        return AgentState(last.position, last.direction, new_color, last.charge - 1), colors
    return None


class AgentTestCase(unittest.TestCase):
    def test_steps_sequence_and_state_update(self):
        # Rule chain: dummy_rule -> dummy_rule
        node2 = RuleNode(rule=dummy_rule)
        node1 = RuleNode(rule=dummy_rule, next_node=node2)
        agent = Agent(
            position=PointSet([(0, 0)]),
            direction="none",
            label="A",
            node=node1,
            colors=iter([1, 2, 3]),
            charge=2
        )
        states = list(agent.steps(PointSet(), {}))
        self.assertEqual(2, len(states))
        self.assertEqual(PointSet([(0, 2)]), agent.position)
        self.assertEqual("right", agent.direction)
        self.assertEqual(3, agent.color)
        self.assertEqual(0, agent.charge)
        self.assertEqual(2, len(agent.history))  # NOTE: Is this correct? Should the initial state be included?

    @unittest.skip("Specifying None rules is not supported")
    def test_steps_none_node(self):
        agent = Agent(
            position=PointSet([(0, 0)]),
            direction="none",
            label="A",
            node=None,
            colors=iter([1]),
            charge=1
        )
        states = list(agent.steps(PointSet(), {}))
        self.assertEqual(states, [])

    def test_steps_alternative_node(self):
        alt_node = RuleNode(rule=dummy_rule)
        # Main rule returns None
        main_node = RuleNode(rule=lambda *args, **kwargs: None, alternative_node=alt_node)

        agent = Agent(
            position=PointSet([(0, 0)]),
            direction="none",
            label="A",
            node=main_node,
            colors=iter([5, 2]),
            charge=1
        )

        states = list(agent.steps(PointSet(), {}))
        self.assertEqual(1, len(states))
        self.assertEqual(PointSet([(0, 1)]), agent.position)
        self.assertEqual("right", agent.direction)
        self.assertEqual(2, agent.color)
        self.assertEqual(0, agent.charge)

    def test_steps_collision_and_color_iterator(self):
        node = RuleNode(rule=collision_rule)
        agent = Agent(
            position=PointSet([(1, 2)]),
            direction="down",
            label="B",
            node=node,
            colors=iter([7, 8]),
            charge=1
        )

        states = list(
            agent.steps(PointSet([(2, 2)]), {(2, 2): AgentState(PointSet([(2, 2)]), "none", 0, 1)})
        )
        self.assertEqual(1, len(states))
        self.assertEqual(8, agent.color)
        self.assertEqual(0, agent.charge)
        self.assertEqual("down", agent.direction)
        self.assertEqual(agent.position, states[0].position)
