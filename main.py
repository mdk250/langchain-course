from typing import List
from dotenv import load_dotenv
from langchain_classic.tools import Tool, tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers import ReActSingleInputOutputParser
from prompttemplate import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS

load_dotenv()


@tool
def get_text_length(text:str)-> int:
    print(f"get_text_length enter with {text=}")
    """Returns the length of a text by characters"""
    text = text.strip("'\n").strip('"')

    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name:str)->Tool:
    for tool in tools:
        if tool.name == tool_name:
            return tool
    
    raise ValueError(f"Tool with name {tool_name} not found")
 
if __name__ == "__main__":
    tools = [get_text_length]

    prompt = PromptTemplate.from_template(REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS).partial(tools=render_text_description(tools), tool_names=", ".join([t.name for t in tools]))

    llm = ChatOpenAI(temperature=0, stop=["\nObservation"])
    agent = {"input": lambda x:x["input"]} | prompt | llm | ReActSingleInputOutputParser() 

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({"input": "What is the text length of 'DOG' in characters?"})
    print(agent_step)

    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use = find_tool_by_name(tools, tool_name)
        tool_input = agent_step.tool_input

        observation = tool_to_use.func(str(tool_input))
        print(f"{observation=}")