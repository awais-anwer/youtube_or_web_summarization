import validators # validators -> used to validate the URL
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

## Streamlit APP
st.set_page_config(page_title="LangChain: Summarize Text From YT or Website", page_icon="🦜")
st.title("🦜 LangChain: Summarize Text From YT or Website")
st.subheader('Summarize URL')

import os
from dotenv import load_dotenv
# laod environment variables
load_dotenv()

# LLM model
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model="Llama-3.1-8b-Instant")

# Prompt 
prompt_template="""
Provide a summary of the following content in 400 words

Content: {text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

generic_url=st.text_input("URL", label_visibility="collapsed")

if st.button("Summarize the Content"):
    ## Validate all the inputs
    if not generic_url.strip():
        st.error("Please provide the URL")
    elif not validators.url(generic_url):
        st.error("Please provide a valid Url. It may be a Youtube video url or website url")
    else:
        try:
            with st.spinner("Waiting.."):
                ## loading data
                if "youtube.com" in generic_url:
                    loader = YoutubeLoader.from_youtube_url(generic_url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(urls=[generic_url], ssl_verify=False, 
                                                   headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"})
                docs = loader.load()

                ## Chain for summarization
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
        except Exception as e:
            st.exception(f"Exception:{e}")
