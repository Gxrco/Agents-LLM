from langchain import hub
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_csv_agent

def create_python_agent() -> AgentExecutor:
    instructions = """You are an agent designed to write and execute python code to answer questions.
    You have access to a python REPL, which you can use to execute python code.
    You have qrcode package installed.
    If you get an error, debug your code and try again.
    Only use the output of your code to answer the question.
    You might know the answer without running any code, but you should still run the code to get the answer.
    If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
    """
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=instructions)
    tools = [PythonREPLTool()]
    python_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools,
    )
    return AgentExecutor(agent=python_agent, tools=tools, verbose=True)

def create_csv_agents() -> tuple[AgentExecutor, ...]:
    fortune_2000_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="data/fortune_2000_in_2021.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    hbo_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="data/HBO_titles.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    gold_price_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="data/gold_price.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    iq_level_agent_executor = create_csv_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-4"),
        path="data/IQ_level.csv",
        verbose=True,
        allow_dangerous_code=True,
    )

    return (
        fortune_2000_agent_executor,
        hbo_agent_executor,
        gold_price_agent_executor,
        iq_level_agent_executor,
    )

def create_grand_agent(tools) -> AgentExecutor:
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions="")
    grand_agent = create_react_agent(
        prompt=prompt,
        llm=ChatOpenAI(temperature=0, model="gpt-4-turbo"),
        tools=tools
    )
    return grand_agent