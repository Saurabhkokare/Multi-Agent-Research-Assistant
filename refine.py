from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.tools import Tool
from langchain.agents import initialize_agent, AgentType
import os
from dotenv import load_dotenv
load_dotenv()

groq_api_key=os.environ['GROQ_API_KEY']

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Define the Refinement Prompt
refinement_prompt = PromptTemplate(
    input_variables=["summary", "critique"],
    template=(
        "Refine the following summary:\n{summary}\n"
        "Based on the critique:\n{critique}\n"
        "Ensure clarity, logical flow, and proper citations."
    )
)

# Define the Refinement Chain
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)

# Define the Tool
def refine_summary_tool(summary: str, critique: str) -> str:
    """Refines an academic summary based on critique feedback."""
    return refinement_chain.run({"summary": summary, "critique": critique})

# Register the Tool
tools = [
    Tool(
        name="RefineSummaryTool",
        func=refine_summary_tool,
        description="Refines an academic summary based on provided critique feedback."
    )
]

# Initialize the Agent
refinement_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True
)

# Agent Execution Function
def refine_summary(summary: str, critique: str) -> str:
    """Refines the summary based on critique feedback."""
    # Use the tool directly instead of letting the agent interpret plain text
    return refine_summary_tool(summary, critique)

# Example Usage
if __name__ == "__main__":
    summary_text = """
    The study explores the use of machine learning in predicting ADME properties of drugs. 
    It highlights the potential of neural networks to outperform traditional statistical models.
    """
    critique_text = """
    While the summary captures the core idea, it lacks specific details on methodologies and evaluation metrics. 
    Including numerical performance benchmarks would improve clarity.
    """
    refined = refine_summary(summary_text, critique_text)
    print("\nRefined Summary:\n", refined)
