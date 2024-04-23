import os
import streamlit as st
import pickle
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from langchain.embeddings import HuggingFaceEmbeddings
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader

from langchain.vectorstores import FAISS
from winmagic import magic
from langchain.vectorstores import FAISS
from transformers import AutoTokenizer,pipeline,AutoModelForQuestionAnswering
from langchain import HuggingFacePipeline
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
model_name ='google/flan-t5-base'
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
model_name ='google/flan-t5-base'
tokenizer = AutoTokenizer.from_pretrained(model_name,padding=True,truncation=True,max_length=512)
question_ans = pipeline("text2text-generation",model = model_name,tokenizer = tokenizer,return_tensors="pt")
llm = HuggingFacePipeline(pipeline=question_ans,model_kwargs ={"temperature":0.7, "max_length": 512})
if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000,
        chunk_overlap=150
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")
    modelPath = 'sentence-transformers/all-MiniLM-l6-v2'
    model_kwargs = {'device':'cuda'}
    encode_kwargs = { 'normalize_embeddings':False}
    embeddings = HuggingFaceEmbeddings(model_name = modelPath,model_kwargs=model_kwargs,encode_kwargs= encode_kwargs)
    vectorindex = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex, f)
def paraphrase_question_with_context(question, context, model_name="google/flan-t5-base"):
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Combine the question and context into a single string
    input_text = f"question: {question} context: {context}"
    
    # Encode the text input
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    # Generate a paraphrased response
    output_ids = model.generate(input_ids, max_length=80, num_beams=5, early_stopping=True)
    paraphrased_answers =[]
    # Decode the generated ids to get the text
    for i in output_ids:
        paraphrased_answers.append(tokenizer.decode(i, skip_special_tokens=True))
    str(paraphrased_answers)
    return paraphrased_answers
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever(search_kwargs={"k":4})
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
            docs_ret = retriever.get_relevant_documents(query)
            ans =docs_ret[0].page_content
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(paraphrase_question_with_context(query, ans)[0])