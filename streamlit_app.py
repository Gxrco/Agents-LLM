import streamlit as st
from agents import create_python_agent, create_csv_agents, create_grand_agent
from langchain_core.tools import Tool
from langchain.agents import AgentExecutor
from typing import Any

python_agent_executor = create_python_agent()
fortune_2000_agent_executor, hbo_agent_executor, gold_price_agent_executor, iq_level_agent_executor = create_csv_agents()

def python_agent_executor_wrapper(original_prompt: str) -> dict[str, Any]:
    return python_agent_executor.invoke({"input": original_prompt})

tools = [
    Tool(name="Python Agent", func=python_agent_executor_wrapper,
         description="""Useful when you need to transform natural language to python and execute the python code,
          returning the results of the code execution DOES NOT ACCEPT CODE AS INPUT"""),
    Tool(name="Fortune 2000 Info Agent", func=fortune_2000_agent_executor.invoke,
         description="""Useful when you need to answer questions over fortune_2000_in_2021.csv, specifically questions about Fortune 2000 companies and their attributes.
         Takes an input the entire question and returns the answer after running pandas calculations"""),
    Tool(name="HBO Titles Info Agent", func=hbo_agent_executor.invoke,
         description="""Useful when you need to answer questions over HBO_titles.csv, specifically questions about HBO titles and their attributes.
         Takes an input the entire question and returns the answer after running pandas calculations"""),
    Tool(name="Gold Price Info Agent", func=gold_price_agent_executor.invoke,
         description="""Useful when you need to answer questions over gold_price.csv, specifically questions about gold prices and their attributes.
         Takes an input the entire question and returns the answer after running pandas calculations"""),
    Tool(name="IQ Level Info Agent", func=iq_level_agent_executor.invoke,
         description="""Useful when you need to answer questions over IQ_level.csv, specifically questions about IQ levels for country.
         Takes an input the entire question and returns the answer after running pandas calculations"""),
]

grand_agent_executor = AgentExecutor(
    agent=create_grand_agent(tools),
    tools=tools,
    verbose=True,
)

st.markdown(
    """
    <style>
    h1 {
        text-align: center;
        color: #E38E49;
    }
    .st-emotion-cache-13k62yr {
        background-color: #0A3981;
    }
    .stAppHeader {
        background-color: #0A3981;
        display: none;
        height: 1px;
    }
    .stButton>button {
        background-color: #E38E49;
        color: black;
        border: 2px solid #1F509A;
    }
    .stTextInput>div>div>input {
        background-color: #D4EBF8;
        color: black;
    }
    .stHeader {
        color: #E38E49;
    }
    .stSelectbox>div>div>div {
        background-color: #D4EBF8;
        color: black;
    }
    .stSelectbox>div>div>div>div>div {
        background-color: #D4EBF8;
        color: black;
    }
    .stSelectbox>div>div>div>div>ul {
        background-color: #D4EBF8;
    }
    .stSelectbox>div>div>div>div>ul>li {
        color: black;
    }
    div[role="listbox"] ul {
    background-color: #D4EBF8;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown(
    """
    # Python Agents - Task Performer Application

    ### Description
    This application allows you to interact with various agents to perform different tasks such as generating QR codes, calculating factorials, and answering questions based on CSV data.

    ### Instructions
    1. **Python Agent Job Selection**: Choose a job from the dropdown and click the "Execute Python Job" button to see the result.
    2. **Ask a Question**: Enter your question in the text input field and click the "Execute Question" button to get an answer from the grand agent executor.

    ### CSV Contents
    The CSV files contain the following information:
    - **fortune_2000_in_2021.csv**: Fortune 2000 companies and their values in the market.
    - **HBO_titles.csv**: HBO titles scores based on user ratings.
    - **gold_price.csv**: Gold prices, all you need to know about the price gold over time.
    - **IQ_level.csv**: IQ levels classified by country.
    """
)

# Job selection for Python Agent
st.header("Python Agent Job Selection")
job_options = [
    "Generate a QR code for a given URL",
    "Calculate the factorial of a number",
    "Generate the code in Python"
]
selected_job = st.selectbox("Select a job for the Python Agent:", job_options)
if st.button("Execute Python Job"):
    if selected_job == "Generate a QR code for a given URL":
        result = python_agent_executor_wrapper("Generate a QR code for https://www.google.com")
    elif selected_job == "Calculate the factorial of a number":
        result = python_agent_executor_wrapper("Calculate the factorial of 5")
    elif selected_job == "Generate the code in Python":
        result = python_agent_executor_wrapper("Generate the code to build a calculator in Python")
    st.write(result["output"])

st.header("Ask a Question")
question = st.text_input("Enter your question here:")
if st.button("Execute Question"):
    result = grand_agent_executor.invoke({"input": question})
    st.write(result["output"])