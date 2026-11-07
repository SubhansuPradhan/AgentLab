"""
Every agent in AgentLab follows the same shape: it takes the shared
LangGraph state, does its one job, and returns a dict of updates to
merge back into that state. Inheriting from BaseAgent keeps that
contract consistent so the graph in graph/workflow.py doesn't need
special-case handling per agent.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseAgent(ABC):
    """
    Subclass this for every new agent. Implement `run`, nothing else
    is required. Keep each agent's `run` method doing exactly one job,
    if you find yourself doing two unrelated things in one agent,
    that's a sign it should be split into two agents.
    """

    name: str = "base_agent"

    @abstractmethod
    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Read whatever this agent needs from `state`, do the work,
        return a dict containing only the keys this agent adds or
        changes. LangGraph merges this back into the full state,
        so don't return the whole state object, just the diff.
        """
        raise NotImplementedError

    def log(self, message: str) -> None:
        print(f"[{self.name}] {message}")
