# Importing necessary libraries
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
api_key = os.getenv('OPENAI_API_KEY')

# Loading files
markdown_path = "the_fall_of_anakin_skywalker.md"
loader = UnstructuredMarkdownLoader(markdown_path)

doc = loader.load()
assert len(doc) == 1
assert isinstance(doc[0], Document)
markdown_content = doc[0].page_content
print(markdown_content[:250])

# Spliting text into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(doc)
len(all_splits)

# Storing documents in Chroma
vectorstore = Chroma.from_documents(
    documents=all_splits, 
    embedding=OpenAIEmbeddings()
)

# Creating the retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3})

# Generating a response
llm = ChatOpenAI(model="gpt-4o-mini") # Create ChatOpenAI instance

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use three sentences maximum and keep the answer as concise as possible.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
custom_rag_prompt = PromptTemplate.from_template(template)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | StrOutputParser()
)

try:
    response = rag_chain.invoke("How did Anakin become Darth Vader?")
    print(response)
except Exception as e:
    print(f"An error occurred: {e}")