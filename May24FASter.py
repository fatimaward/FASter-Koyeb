from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from pdf2image import convert_from_path
import pytesseract
from langchain_core.runnables import RunnableConfig
from PIL import Image
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts import HumanMessagePromptTemplate
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import nest_asyncio
import pandas as pd
from langsmith import Client
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.smith import RunEvalConfig, run_on_dataset
import re

os.environ['OPENAI_API_KEY'] = "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"
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

date1_schema = ResponseSchema(name="Last Periodic Review Date",
                            description="This is the date the document was issued")

date2_schema = ResponseSchema(name="Next Periodic Review Date",
                               description="This is the date the document will be reviewed")

response_schemas = [date1_schema, date2_schema]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

wd = os.getcwd()
# splitter = CharacterTextSplitter(
#       separator="\n\n",
#       chunk_size=400,
#       chunk_overlap=10,
#       length_function=len,
#       is_separator_regex=False,
#   )
# text = ""
# texts = []
# vectorstore_new = FAISS.from_texts([''], OpenAIEmbeddings())

# meta_template = """Use the {text}. Give a summary, showing all the below information. Output the info in the order presented.
# Extract the following information, in this style:

# Last Periodic Review Date (OR Publication Date):
# Next Periodic Review Date:
# Owner:
# Implementation Office(s) (IF PRESENT):

# Format your output according to this: {format_instructions}
# """
# llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
# meta_prompt = PromptTemplate(input_variables=['text', 'format_instructions'], template=meta_template)
# meta_chain = LLMChain(llm=llm, prompt=meta_prompt)
# meta_df = pd.DataFrame()

# txt_files = r'C:\Users\user\Desktop\FASter\FASterTextsMay2'

# for file in os.listdir(txt_files):
#   if file.endswith('txt'):
#     with open(os.path.join(txt_files, file), 'r', encoding='utf-8') as f:
#       filename = file.split('.', 1)[0]
#       text = f.read()
#       loader = TextLoader(os.path.join(r'C:\Users\user\Desktop\FASter\FASterTextsMay2', file), encoding='UTF-8')
#       documents = loader.load()
#       splits = splitter.split_documents(documents)
#       meta = meta_chain.invoke({"text": str(splits[0]), "format_instructions": format_instructions})
#       meta = str(meta)
#       meta = eval(meta)
#       meta_as_dict = output_parser.parse(meta['text'])
#       new_meta = pd.DataFrame(meta_as_dict, index=[0])
#       meta_df = pd.concat([new_meta, meta_df], ignore_index=True)
#       splits = ["Policy Name: " + filename + '\n' + meta['text'] + str(element) for element in splits]
#       new_vectorstore = FAISS.from_texts(splits, OpenAIEmbeddings())
#       vectorstore_new.merge_from(new_vectorstore)

# embeddings = OpenAIEmbeddings()
# vectorstore_og = FAISS.load_local(r'C:\Users\user\Desktop\FASter\FAStervs\FAStervsMay2', embeddings, allow_dangerous_deserialization=True)
# vectorstore_og.merge_from(vectorstore_new)
# vectorstore_og.save_local(r'C:\Users\user\Desktop\FASter\FAStervs\FAStervsMay4')

os.environ['OPENAI_API_KEY'] = "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"

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-0125")

embeddings = OpenAIEmbeddings()

wd = os.getcwd()

vectorstore = FAISS.load_local('FAStervsMay24E', embeddings, allow_dangerous_deserialization=True)

sp_data = pd.read_csv('May2Query.csv', encoding='cp1252')

sp_data = sp_data.rename(columns={'Name': 'Policy Name'})

retriever = vectorstore.as_retriever()

from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Sequence, Annotated, Optional
import operator
from langchain_core.messages import BaseMessage

class GraphState(TypedDict):
  question: Optional[str] = None
  answer: Optional[str] = None
  sources: Optional[list] = None
  link: Optional[str] = None
  policy_name: Optional[str] = None
  date1: Optional[str] = None
  date2: Optional[str] = None

workflowt = StateGraph(GraphState)

answer_schema = ResponseSchema(name="Answer",
                             description="This is your answer")

policy_name_schema = ResponseSchema(name="Policy Name",
                             description="This is the policy name. When you find it, output it EXACTLY as is.")

date1_schema = ResponseSchema(name="Last Periodic Review Date",
                            description="This is the date the document was issued")

date2_schema = ResponseSchema(name="Next Periodic Review Date",
                               description="This is the date the document will be reviewed")
response_schemas_graph = [answer_schema, policy_name_schema, date1_schema, date2_schema]

output_parser_graph = StructuredOutputParser.from_response_schemas(response_schemas_graph)
format_instructions_graph = output_parser_graph.get_format_instructions()

template = [HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=['context', 'question'], template="You are an assistant for question-answering tasks related to information at a university faculty. Use the following pieces of retrieved context to answer the question. Be concise and informative. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.\nQuestion: {question} \nContext: {context} \nAnswer:"))]
prompt = ChatPromptTemplate(messages=template, input_variables=['context', 'question'])
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)
def retrieve(state):
  question = state.get('question').strip()
  rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
  answer = rag_chain.invoke(question)
  new_str = question + "Faculty directory."
  docs = retriever.invoke(new_str, config=RunnableConfig(metadata=dict(policy_name="Faculty Directory-Not a policy")))
  sources = []
  for doc in docs:
    data_string = doc.page_content
    date1_match = re.search(r'"Last Periodic Review Date": "([^"]+)"', data_string)
    date1 = date1_match.group(1) if date1_match else None
    date2_match = re.search(r'"Next Periodic Review Date": "([^"]+)"', data_string)
    date2 = date2_match.group(1) if date2_match else None
    index_policy_name = data_string.find("Policy Name: ")
    if index_policy_name != -1:
      index_newline = data_string.find('\n', index_policy_name)
      if index_newline != -1:
        policy_name = data_string[index_policy_name+len("Policy Name: "):index_newline]
    page_content_match = re.search(r"```page_content='([^']+)'", data_string)
    page_content = page_content_match.group(1) if page_content_match else None
    sources.append({f"Retrieved from {policy_name}": page_content})
  return {"answer": answer, "sources": sources, "policy_name": policy_name, "date1": date1, "date2": date2}

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

def find_link(state):
  sources = state.get('sources')
  for element in sources:
    key = element.keys()
    policy_name = state.get('policy_name')
    policy_name_pdf = policy_name.lower() + '.pdf'
    policy_name_docx = policy_name.lower() + '.docx'
    sp_data['Policy Name'] = sp_data['Policy Name'].str.lower()
    if not sp_data.loc[sp_data['Policy Name'] == policy_name_pdf].empty:
       link_df = sp_data.loc[sp_data['Policy Name'] == policy_name_pdf]
    elif not sp_data.loc[sp_data['Policy Name'] == policy_name_docx].empty:
      link_df = sp_data.loc[sp_data['Policy Name'] == policy_name_docx]
    else:
       link_df = pd.DataFrame({'Path': [], 'Full Link': ''})
    if not link_df['Path'].isnull().all():
      link = 'https://mailaub.sharepoint.com/' + link_df['Path'].str.replace(' ', '%20', regex=True) + '\\' + link_df['Policy Name'].str.replace(' ', '%20', regex=True)
    else:
       link = link_df['Full Link'].str.replace(' ', ' ', regex=True)
  return {"link": link}

workflowt.add_node("retrieve", retrieve)
workflowt.add_node("find_link", find_link)

workflowt.set_entry_point("retrieve")
workflowt.add_edge("retrieve", "find_link")
workflowt.add_edge("find_link", END)

app = workflowt.compile()

def get_the_answer(question):
    global link_to_get
    output = app.invoke({"question": question})
    link_string = str(output['link'])
    start_of_link = link_string.find('https')
    end_of_link = link_string.find('dtype')
    end_of_link_b = link_string.find('Name:')
    link_to_get = link_string[start_of_link:]
    end_index = min(end_of_link, end_of_link_b) if end_of_link != -1 and end_of_link_b != -1 else max(end_of_link, end_of_link_b)
    if end_index != -1:
        link_to_get = link_string[start_of_link:end_index]
    elif start_of_link != -1:
        link_to_get = link_string[start_of_link:]
    answer = str(output['answer']) + '\n' + "This policy was published on " + str(output['date1']) + ".\n" + "This policy's next review date is " + str(output['date2']) + '.'
    link_base = ' Retrieved from ' + str(output['policy_name']) + ": " + '<a href="' + link_to_get + '" target="_blank">Click here</a>'
    output = answer + link_base
    return output

flask_app = Flask(__name__)
CORS(flask_app, supports_credentials=True, origins='http://localhost:8000')
@flask_app.route('/')
def index():
    return 'Check'
@flask_app.route('/api/generate-response-faster', methods=['POST'])
def generate_response():
    data = request.json
    question = data.get('message')
    main_answer = get_the_answer(question)
    return jsonify({'response': main_answer})
if __name__ == '__main__':
    flask_app.run(debug=True)