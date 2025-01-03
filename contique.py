from langchain.tools import TavilySearchResults
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_groq import ChatGroq
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
groq_api_key = os.environ['GROQ_API_KEY']

def contique_summary():
# Initialize LLM

    llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Define Tools
    search_tool = TavilySearchResults(tavily_api_key="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")
    tools = [
        Tool(
            name="Academic Search",
            func=search_tool.run,
            description="Use this tool to search for academic papers or related information based on a given summary."
        )
    ]

# Define the Critique Prompt
    critique_prompt = PromptTemplate(
        input_variables=["summary", "related_papers"],
        template=(
            "Given the summary:\n{summary}\n"
            "And the related papers:\n{related_papers}\n"
            "Provide a detailed critique highlighting strengths, weaknesses, and missing insights."
        )
    )

    # Define the Critique Chain
    critique_chain = LLMChain(llm=llm, prompt=critique_prompt)
    return critique_chain
# Initialize the Agent
from langchain.agents import initialize_agent

summary_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent Execution Function
def critique_summary(summary: str) -> str:
    """Critiques the summary using external academic sources."""
    # Fetch related academic papers
    related_papers = summary_agent.run(f"Find academic papers related to the following summary: {summary}")
    
    # Generate critique based on summary and papers
    critique = critique_chain.run(summary=summary, related_papers=related_papers)
    return critique

# Example Usage
summary_text = "Machine learning models are being increasingly used to predict ADME properties in drug discovery."
critique = critique_summary(summary_text)
print("Critique:\n", critique)
