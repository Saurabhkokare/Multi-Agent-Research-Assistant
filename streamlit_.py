import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain_groq import ChatGroq
import fitz
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()

gr_api_key=os.environ['GROQ_API_KEY']

# Initialize LLM
llm = ChatGroq(model="llama3-8b-8192", api_key=gr_api_key)

# Define prompts
summary_prompt = PromptTemplate(
    input_variables=["content"],
    template="Summarize the following academic paper into key arguments, methodologies, and conclusions:\n\n{content}"
)
summary_chain = LLMChain(llm=llm, prompt=summary_prompt)

critique_prompt = PromptTemplate(
    input_variables=["summary", "related_content"],
    template=(
        "Critique the following summary based on the additional content:\nSummary: {summary}\n\nRelated Content: {related_content}\n"
        "Identify discrepancies, missing insights, and provide strengths and weaknesses."
    )
)
critique_chain = LLMChain(llm=llm, prompt=critique_prompt)

refinement_prompt = PromptTemplate(
    input_variables=["summary", "critique"],
    template=(
        "Refine the following summary using the critique provided:\nSummary: {summary}\nCritique: {critique}\n"
        "Ensure clarity, logical flow, and add citations where applicable."
    )
)
refinement_chain = LLMChain(llm=llm, prompt=refinement_prompt)

# Tools
tools = [
    Tool(
        name="SummarizePaper",
        func=summary_chain.run,
        description="Use this tool to summarize academic papers into key arguments, methodologies, and conclusions."
    ),
    Tool(
        name="CritiquePaper",
        func=critique_chain.run,
        description="Use this tool to critique summaries based on additional related content."
    ),
    Tool(
        name="RefineSummary",
        func=refinement_chain.run,
        description="Use this tool to refine a summary based on critique feedback."
    )
]

multi_agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Helper Functions
def load_document(file_path: str):
    """Load text from a PDF or text file."""
    if file_path.endswith('.pdf'):
        doc = fitz.open(file_path)
        text = "".join([page.get_text() for page in doc])
        return text
    else:
        with open(file_path, 'r') as file:
            return file.read()


def split_text_into_chunks(text, max_length=3000):
    """Split text into smaller chunks."""
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]


def orchestrate_workflow(paper_text):
    """Run the summarization, critique, and refinement workflow."""
    paper_chunks = split_text_into_chunks(paper_text)

    summaries = []
    for chunk in paper_chunks:
        summary = summary_chain.run({"content": chunk})
        summaries.append(summary)

    full_summary = " ".join(summaries)
    related_content = "Additional related content that can help with critique."

    critique = critique_chain.run({"summary": full_summary, "related_content": related_content})
    refined_summary = refinement_chain.run({"summary": full_summary, "critique": critique})

    return full_summary, critique, refined_summary



st.title("📚 AI Assistant for Summarizer , Critique & Refined Academic Paper ")
st.write("Upload an academic paper (PDF or Text).")

uploaded_file = st.file_uploader("Upload your paper", type=["pdf", "txt"])

if uploaded_file:
    file_path = f"./uploaded_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.success(f"File {uploaded_file.name} uploaded successfully!")

    if st.button("📝 Process Paper"):
        with st.spinner("Processing... This may take a moment."):
            paper_content = load_document(file_path)
            summary, critique, refined_summary = orchestrate_workflow(paper_content)
        
        st.subheader("📝 Summary")
        st.write(summary)

        st.subheader("🧐 Critique")
        st.write(critique)

        st.subheader("✅ Refined Summary")
        st.write(refined_summary)
