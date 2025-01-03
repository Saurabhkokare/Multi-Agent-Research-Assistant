from workflow import graph, initial_state

def main():
    # Example academic paper content
    paper_text = """
                Building an AI agent that leverages realtime online information is not a simple task. Scraping doesn't scale and requires expertise to refine, current search engine APIs don't provide explicit information to queries but simply potential related articles (which are not always related), and are not very customziable for AI agent needs. This is why we're excited to introduce the first search engine for AI agents - Tavily Search API.

Tavily Search API is a search engine optimized for LLMs, aimed at efficient, quick and persistent search results. Unlike other search APIs such as Serp or Google, Tavily focuses on optimizing search for AI developers and autonomous AI agents. We take care of all the burden of searching, scraping, filtering and extracting the most relevant information from online sources. All in a single API call!

To try the API in action, you can now use our hosted version on our API Playground.

If you're an AI developer looking to integrate your application with our API, or seek increased API limits, please reach out!


    """
    
    # Initialize state as a dictionary
    state = initial_state(paper_text)
    
    # Execute the workflow
    final_state = graph.invoke(state)
    
    print("\nFinal Refined Summary:")
    print(final_state["refined_summary"])

if __name__ == "__main__":
    main()
