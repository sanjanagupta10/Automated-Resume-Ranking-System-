from dotenv import load_dotenv

load_dotenv()
import os
import streamlit
import pinecone
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.callbacks import get_openai_callback
import pandas as pd
from io import StringIO
import json

streamlit.set_page_config(layout="wide")

streamlit.markdown("<h1 style='text-align: center; color: blue;'>Resume Ranking</h1>", unsafe_allow_html=True)
uploaded_file = streamlit.file_uploader("Choose a file")  ##Upload Job Description as a file
if uploaded_file is not None:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    jobDescription = stringio.read()

if streamlit.button("Search"):

    llm = OpenAI(temperature=0, max_tokens=1024)
    jd_query = """Summarize below and content and show me only the Skills, total years of experience
     and highest qualification in a single row\n\n {job_description}
    """.format(job_description=jobDescription)
    response = llm.invoke(jd_query)

    # Search resumes in pinecone db
    pinecone.init(api_key=os.getenv("api_key"), environment='gcp-starter')

    index = pinecone.Index(os.getenv("pinecone_index"))
    embeddings = OpenAIEmbeddings()
    query_result = embeddings.embed_query(response)
    result = index.query(query_result, top_k=3, include_values=False)

    response_array = []
    # Read document by id(file path) and pass it to OpenAI for summary
    for res in result.matches:
        loader = DirectoryLoader(os.getenv("data_source_path"))
        docs = loader.load()
        for doc in docs:
            if doc.metadata["source"] == res.id:
                query = f"""
                        Notice: Include only top 5 Skills, include 3 roles, include 3 industries, if json values have more than 1 then prepare as csv string
                        Response Json keys are Name, Contact Details, Skills, total years of experience, Highest Qualification, Roles, Industries, Score
                        Provide Name, Contact Details, Skills, Total years of experience, and highest qualification, roles, industries
                        from this document {doc}
                        and append score = {res.score} at the last 
                        and response should be in JSON format
                        """
                with get_openai_callback() as cb:
                    response = llm.invoke(query)
                    r1 = json.loads(response)
                    response_array.append(r1)
    array_json = json.dumps(response_array)
    r = json.loads(array_json)
    df = pd.json_normalize(r, meta=["Col1", "col2", "col3", "Col4", "col5", "col6"])
    streamlit.table(df)
    with streamlit.expander(":blue[**Cost Statistics**]"):
        statsCol1, statsCol2, statsCol3, statsCol4, statsCol5 = streamlit.columns(5, gap='small')
        statsCol1.metric(label=":grey[*No of Request*]", value=f"{cb.successful_requests}")
        statsCol2.metric(label=":grey[*Prompt Tokens*]", value=f"{cb.prompt_tokens}")
        statsCol3.metric(label=":grey[*Completion Tokens*]", value=f"{cb.completion_tokens}")
        statsCol4.metric(label=":grey[*Total Tokens*]", value=f"{cb.total_tokens}")
        statsCol5.metric(label=":grey[*Total Cost (USD)*]", value=f"${cb.total_cost:.4f}")
