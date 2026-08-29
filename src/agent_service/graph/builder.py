from langgraph.graph import START, StateGraph
from langgraph.types import interrupt

from ..mcp_client import MCPCaller
from ..model_client import ModelClient
from ..state import MatterDraft
from .nodes_intake import make_intake_nodes


def build_graph(model: ModelClient, mcp: MCPCaller, checkpointer):
    builder = StateGraph(MatterDraft)
    nodes: dict = {}
    nodes.update(make_intake_nodes(model, mcp))
    # Tasks 10-12 add: make_risk_nodes, make_setup_nodes, make_review_nodes
    try:
        from .nodes_risk import make_risk_nodes
        nodes.update(make_risk_nodes(model, mcp))
    except ImportError:
        pass
    try:
        from .nodes_setup import make_setup_nodes
        nodes.update(make_setup_nodes(model, mcp))
    except ImportError:
        pass
    try:
        from .nodes_review import make_review_nodes
        nodes.update(make_review_nodes(model, mcp))
    except ImportError:
        pass

    # Placeholder nodes for stages not yet implemented (Tasks 10-12) so that
    # Command(goto=...) targets from earlier stages always resolve. Each
    # placeholder just interrupts; it is superseded once the real node module
    # lands and is registered above.
    for placeholder_name in ("review", "submit"):
        if placeholder_name not in nodes:
            async def _placeholder(state: MatterDraft):
                interrupt({})
            nodes[placeholder_name] = _placeholder

    for name, fn in nodes.items():
        builder.add_node(name, fn)
    builder.add_edge(START, "welcome")
    return builder.compile(checkpointer=checkpointer)
