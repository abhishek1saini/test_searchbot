import os
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import streamlit as st


pinecone.init(
            api_key=st.secrets["PINECONE_API_KEY_SEARCH_BOT"],  #from streamlit
            environment=st.secrets["PINECONE_ENV_SEARCH_BOT"]
        )
index_name = "pdf-test" # name of your pinecone index to be used
index = pinecone.Index(index_name)
index_stats_response = index.describe_index_stats()
print(index_stats_response)

llm = ChatOpenAI()
embeddings = OpenAIEmbeddings()
doc_db = Pinecone.from_existing_index(index_name, embeddings)
def retrieval_response(query):
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type= 'stuff',    #Others are ie. mapreduce etc
        retriever = doc_db.as_retriever()
    )
    query = query
    output = qa.run(query)
    return output

#For streamlit
def main():
    st.title("Sylva Docuemnt QA tool powered by LLM & Pinecone")
    text_input = st.text_input("Please ask your query...")
    if st.button("Ask Query"):
        if len(text_input) > 0:
            st.info("Your Query is: " + text_input)
            answer = retrieval_response(text_input)
            st.success(answer)

if __name__ == "__main__":
    main()
