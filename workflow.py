from langgraph.graph import StateGraph, END, START
from sum_agent import summarize_paper
from contique import critique_summary
from refine import refine_summary


# Define the initial state structure
def initial_state(paper_text: str):
    return {
        "paper_text": paper_text,
        "summary": None,
        "critique": None,
        "refined_summary": None
    }

# Agents
def summarization_agent(state: dict):
    state["summary"] = f"Summary of: {state['paper_text']}"
    return state

def critique_agent(state: dict):
    state["critique"] = f"Critique based on summary: {state['summary']}"
    return state

def refinement_agent(state: dict):
    state["refined_summary"] = f"Refined Summary: {state['summary']} | {state['critique']}"
    return state

# Build the workflow graph
workflow = StateGraph(dict)

# Add Nodes
workflow.add_node("summarization", summarize_paper)
workflow.add_node("critique", critique_summary)
workflow.add_node("refinement", refine_summary)

# Define edges
workflow.add_edge(START, "summarization")
workflow.add_edge("summarization", "critique")
workflow.add_edge("critique", "refinement")
workflow.add_edge("refinement", END)

# Compile the graph
graph = workflow.compile()
