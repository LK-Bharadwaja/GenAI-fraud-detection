from langgraph.graph import StateGraph
from typing import TypedDict


# =========================
# 1. STATE DEFINITION
# =========================
class RiskState(TypedDict):
    entity_id: str
    risk_level: str
    decision: str


# =========================
# 2. AGENT FUNCTIONS
# =========================
def router(state: RiskState) -> RiskState:
    """
    Routes execution based on risk level.
    """
    return state


def low_risk_agent(state: RiskState) -> RiskState:
    state["decision"] = (
        f"Entity {state['entity_id']} classified as LOW risk. "
        "Monitor periodically."
    )
    return state


def medium_risk_agent(state: RiskState) -> RiskState:
    state["decision"] = (
        f"Entity {state['entity_id']} classified as MEDIUM risk. "
        "Flag for analyst review."
    )
    return state


def high_risk_agent(state: RiskState) -> RiskState:
    state["decision"] = (
        f"Entity {state['entity_id']} classified as HIGH risk. "
        "Trigger immediate investigation."
    )
    return state


# =========================
# 3. GRAPH BUILDER
# =========================
def build_agent_graph():
    graph = StateGraph(RiskState)

    # ✅ ADD NODES FIRST
    graph.add_node("router", router)
    graph.add_node("LOW", low_risk_agent)
    graph.add_node("MEDIUM", medium_risk_agent)
    graph.add_node("HIGH", high_risk_agent)

    # ✅ ENTRY POINT
    graph.set_entry_point("router")

    # ✅ CONDITIONAL ROUTING
    graph.add_conditional_edges(
        "router",
        lambda state: state["risk_level"],
        {
            "LOW": "LOW",
            "MEDIUM": "MEDIUM",
            "HIGH": "HIGH",
        },
    )

    return graph.compile()
