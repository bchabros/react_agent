from dotenv import load_dotenv

from langgraph.graph import END
from langchain_core.agents import AgentFinish
from langgraph.graph import StateGraph

from src.state import AgentState
from src.const import ACT, AGENT_REASON
from src.nodes import execute_tools, run_agent_reasoning_engine

load_dotenv()


def should_continue(state: AgentState) -> str:
    """
    :param state: A dictionary representing the state of the agent. It includes the key "agent_outcome" which indicates the agent's final outcome.
    :return: A string "END" if the agent's outcome is an instance of AgentFinish, otherwise returns "ACT".
    """
    if isinstance(state["agent_outcome"], AgentFinish):
        return END
    else:
        return ACT


def setup_react_graph():
    """
    Sets up a reactive graph and its nodes for agent processing.

    Creates a StateGraph with initial nodes and establishes the entry point.
    Adds specific nodes for reasoning and action execution,
    and configures the transitions between these nodes, including conditional edges.

    :return: Initialized StateGraph instance with configured nodes and edges.
    """
    graph = StateGraph(AgentState)

    graph.add_node(AGENT_REASON, run_agent_reasoning_engine)
    graph.set_entry_point(AGENT_REASON)
    graph.add_node(ACT, execute_tools)
    graph.add_conditional_edges(AGENT_REASON, should_continue)
    graph.add_edge(ACT, AGENT_REASON)

    return graph


flow = setup_react_graph()
app = flow.compile()
