from langgraph.graph import START, StateGraph

from ..mcp_client import MCPCaller
from ..model_client import ModelClient
from ..state import MatterDraft
from .nodes_intake import make_intake_nodes
from .nodes_review import make_review_nodes
from .nodes_risk import make_risk_nodes
from .nodes_setup import make_setup_nodes


def build_graph(model: ModelClient, mcp: MCPCaller, checkpointer):
    builder = StateGraph(MatterDraft)
    nodes: dict = {}
    nodes.update(make_intake_nodes(model, mcp))
    nodes.update(make_risk_nodes(model, mcp))
    nodes.update(make_setup_nodes(model, mcp))
    nodes.update(make_review_nodes(model, mcp))

    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "welcome")
    return builder.compile(checkpointer=checkpointer)
