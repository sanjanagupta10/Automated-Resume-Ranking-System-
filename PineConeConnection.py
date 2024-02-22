from dotenv import load_dotenv

load_dotenv()
import os
import pinecone
from dotenv import load_dotenv
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
import time

load_dotenv()
embeddings = OpenAIEmbeddings()

llm = OpenAI()
pinecone.init(api_key=os.getenv("api_key"), environment='gcp-starter')
index = pinecone.Index(os.getenv("pinecone_index"))

documents = []
loader = DirectoryLoader('DataStore\\Resumes')
docs = loader.load()
prefix_path ="/opt/render/project/src/"
for doc in docs:
    query = """Provide Name, Contact Details, Skills, Total years of experience, and highest qualification,
     different roles performed in different industries response should be in a single table row \n\n {resume}""".format(
        resume=doc)
    response = llm.invoke(query)
    documents.append(response)
    emb = embeddings.embed_documents(documents)
    documents.clear()
    path =(prefix_path + doc.metadata["source"]).replace("\\","/")
    print("path: {}".format(path))
    index.upsert([(path, emb)])
    print("Inserted")
 #   time.sleep(100)
