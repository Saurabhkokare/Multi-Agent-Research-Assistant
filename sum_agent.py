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

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=groq_api_key)

# Define the Summarization Prompt
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template=(
        "Summarize the following academic paper into key arguments, methodologies, and conclusions:\n\n{content}"
    )
)

# Define the Summarization Chain
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

# Define Tools (if any future tools need to be added, they can be listed here)
tools = [
    Tool(
        name="Paper Summarizer",
        func=summary_chain.run,
        description="Use this tool to summarize academic papers into key arguments, methodologies, and conclusions."
    )
]

# Initialize the Agent
from langchain.agents import initialize_agent

summary_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Agent Execution Function
def summarize_paper(paper_text: str) -> str:
    """Summarizes academic content."""
    summary = summary_agent.run(f"Summarize the following academic paper:\n\n{paper_text}")
    return summary

# Example Usage
paper_text = """
The study investigates the application of deep learning models in predicting ADME properties of chemical compounds. 
Methodologies include convolutional neural networks (CNNs) and recurrent neural networks (RNNs). 
The results demonstrate improved accuracy in ADME prediction compared to traditional statistical methods.
"""
summary = summarize_paper(paper_text)
print("Summary:\n", summary)
