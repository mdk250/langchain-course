from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
load_dotenv()


@tool
def search(query: str) -> str:
    """
    Tool that searches the internet
    
    :param query: The query to search for
    :type query: str
    :return: The search result
    :rtype: str
    """
    print(f"Searching for {query}")
    return tavily.search(query=query)

llm = ChatOpenAI(model='gpt-5-nano')
tools = [TavilySearch]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({"messages": HumanMessage(content="search for 3 job openings for an ai engineer in amsterdam using langchain on linkedin and list their details")})
    print(result)

if __name__ == "__main__":
    main()
