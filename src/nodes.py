from dotenv import load_dotenv
from langgraph.prebuilt.tool_executor import ToolExecutor

from src.react import react_agent_runnable, tools
from src.state import AgentState

load_dotenv()


def run_agent_reasoning_engine(state: AgentState):
    """
    :param state: The current state of the agent, typically containing various parameters and context necessary for decision-making processes.
    :return: A dictionary with the outcome of the agent's reasoning process under the key 'agent_outcome'.
    """
    agent_outcome = react_agent_runnable.invoke(state)
    return {"agent_outcome": agent_outcome}


tool_executor = ToolExecutor(tools)


def execute_tools(state: AgentState):
    """
    :param state: The current state of the agent, represented as an AgentState object.
    :return: A dictionary containing intermediate steps taken by the agent,
             which includes the agent action and its corresponding output.
    """
    agent_action = state["agent_outcome"]
    output = tool_executor.invoke(agent_action)
    return {"intermediate_steps": [(agent_action, str(output))]}
