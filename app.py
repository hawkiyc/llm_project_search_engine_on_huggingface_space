#%% Import Libraries

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper, GoogleSerperAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, GoogleSerperRun
from langchain.agents import initialize_agent, AgentType
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler
import os
from pathlib import Path
from dotenv import load_dotenv
import asyncio

#%% Set Async

try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

#%% Get Correct Path

current_dir = Path(__file__).parent.absolute()
doc_dir = os.path.join(current_dir, "research_papers")
db_path = os.path.join(current_dir, "FAISS_DB")

#%% Load API Key

if 'STREAMLIT_PUBLIC_PATH' in os.environ:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    os.environ["HF_TOKEN"] = st.secrets['HUGGINGFACE_TOKEN']
    os.environ["SERPER_API_KEY"] = st.secrets['SERPER_API_KEY']
else:
    load_dotenv()
    groq_api_key = os.getenv('GROQ_API_KEY')
    os.environ["HF_TOKEN"] = os.getenv('HUGGINGFACE_TOKEN')
    os.environ["SERPER_API_KEY"] = os.getenv('SERPER_API_KEY')

#%% API Wrapper and Tools

arxiv_wrapper = ArxivAPIWrapper(top_k_results = 3,doc_content_chars_max = 512)
arxiv_tool = ArxivQueryRun(api_wrapper = arxiv_wrapper)

wiki_wrapper = WikipediaAPIWrapper(top_k_results = 3,doc_content_chars_max = 512)
wiki_tool = WikipediaQueryRun(api_wrapper = wiki_wrapper)

search_wrapper = GoogleSerperAPIWrapper(k = 3)
search_tool = GoogleSerperRun(api_wrapper = search_wrapper,)

tools = [arxiv_tool, wiki_tool, search_tool]

#%% Build Streamlit App

st.write('ChatBot with Search Engine!!!!')

if 'messages' not in st.session_state:
    st.session_state.messages = [
        {'role': 'assistant', 
         'content': 'I am a chatbot with search engine. I am here to answer your question. What do you want to know?'}]

for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

if prompt := st.chat_input(placeholder = 'What is Deep Learning?'):
    st.session_state.messages.append({'role':'user','content':prompt})
    st.chat_message('user').write(prompt)
    
    llm = ChatGroq(
        api_key=groq_api_key, 
        model="llama-3.2-90b-vision-preview",
        streaming=True,
        max_tokens=1024,
        temperature=.7,)
    agent = initialize_agent(
        tools, llm, 
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
        handling_parsing_errors=True,
        max_iterations=50,
        max_execution_time=90)
    
    with st.chat_message('assistant'):
        callback=StreamlitCallbackHandler(st.container(), expand_new_thoughts=True)
        try:
            response=agent.run(st.session_state.messages, callbacks=[callback])
            st.session_state.messages.append({'role': 'assistant', 'content':response})
            st.write(response)
        except ValueError as e:
            st.error(f"An error occurred while parsing the response: {e}")
            st.session_state.messages.append(
                {'role': 'assistant', 
                'content': 'Sorry, I encountered an issue processing your request. Please try asking in a different way.'})
            st.write('Sorry, I encountered an issue processing your request. Please try asking in a different way.')