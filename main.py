from tavily import TavilyClient
import asyncio
import time
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from dotenv import load_dotenv
from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel
from azure.identity import DefaultAzureCredential
#from azure.core.credentials import AzureStaticTokenCredential

import os

load_dotenv()

@tool
def check_weather(location : str) -> str:
    """
    A function that returns the weather of a given location
    """
    return f"The weather is always sunny in {location}"

@tool
def multiply(a : int, b : int) -> int:
    """
    Takes in two parameters a,b

    Multiplys them and returns the result
    """

    return a * b

@tool
def divide(a : int, b : int) -> int:
    """
    Takes in two parameters a,b

    Divides a by b and returns the result
    """

    return a / b

@tool
def add(a : int, b : int) -> int:
    """
    Takes in two parameters a,b

    adds a and b and returns the result
    """

    return a + b

@tool
def subtract(a : int, b : int) -> int:
    """
    Takes in two parameters a,b

    Subtracts b from a and then returns the result
    """

    return a - b

@tool
def search_internet(query : str) -> str:
    """
    A function that takes in a user query

    then utilizes TavilyClient to search the internet for the response
    returns the response to the query
    """

    tavily = TavilyClient(os.getenv("TAVILY_API_KEY"))
    result = tavily.search(query)
    return result['answer']

def main():
    llm = ChatOpenAI(
        model="gpt-5.4",
        temperature=0.1,
        max_tokens=1000,
        timeout=30
    )
    #cred = AzureStaticTokenCredential(os.getenv("AZURE_OPENAI_API_KEY"))
    
    azure_llm = AzureAIOpenAIApiChatModel(
        project_endpoint = os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        model="gpt-4.1-mini",  # e.g., "gpt-4o" or "mistral-large"
        credential=DefaultAzureCredential(),
    )
    agent = create_agent(
        model=llm,
        tools=[check_weather, search_internet, multiply, divide, add, subtract],
        system_prompt="You are a helpful assistant"
    )
    
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Can you solve 3 * 4 + 5 ?"}]}
    )
    print(result)


if __name__ == "__main__":
    main()
    # start_time = time.perf_counter()
    # asyncio.run(main())
    # end_time = time.perf_counter()

    # print(start_time - end_time)