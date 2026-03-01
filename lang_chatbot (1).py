# -*- coding: utf-8 -*-
import streamlit as st

# ✅ ALL Hugging Face imports from langchain_huggingface
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="LangChain Chatbot", page_icon="🤖")
st.title("🤖 Chatbot with Runtime API Key (LangChain + Hugging Face)")

hf_api_key = st.text_input(
    "Enter your Hugging Face API Token",
    type="password"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Type your message..."):
    if not hf_api_key:
        st.error("⚠️ Please enter Hugging Face token.")
    else:
        st.session_state.messages.append(
            {"role": "user", "content": prompt}
        )
        with st.chat_message("user"):
            st.markdown(prompt)

        # ✅ Correct HF endpoint
        llm = HuggingFaceEndpoint(
            repo_id="mistralai/Mistral-7B-Instruct-v0.2",
            huggingfacehub_api_token=hf_api_key,
            temperature=0.7,
            max_new_tokens=512,
        )

        # ✅ Correct chat wrapper
        chat_model = ChatHuggingFace(llm=llm)

        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant."),
            ("user", "{question}")
        ])

        chain = prompt_template | chat_model | StrOutputParser()

        bot_reply = chain.invoke({"question": prompt})

        st.session_state.messages.append(
            {"role": "assistant", "content": bot_reply}
        )
        with st.chat_message("assistant"):
            st.markdown(bot_reply)
