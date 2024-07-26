from dotenv import load_dotenv
import os
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import chromadb as db
from chromadb import Client
from chromadb.config import Settings
from langchain_community.llms import HuggingFaceHub
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from transformers import T5ForConditionalGeneration, T5Tokenizer
import logging

class AnswerOnlyOutputParser(StrOutputParser):
    def parse(self, response):
        if "you do not have access" in response.lower():
            return "You do not have access"
        return response.split("Answer:")[1].strip() if "Answer:" in response else response.strip()

class ChatBot():
    def __init__(self, llm_type="Local (PHI3)", api_key=""):
        load_dotenv()
        self.list_files_in_current_directory()
        self.chroma_client, self.collection = self.initialize_chromadb()
        self.llm_type = llm_type
        self.api_key = api_key
        self.setup_language_model()
        self.setup_langchain()
        self.setup_reranker()

    def list_files_in_current_directory(self):
        current_dir = os.getcwd()
        print(f'Current Directory: {current_dir}')
        with os.scandir(current_dir) as entries:
            for entry in entries:
                if entry.is_file():
                    print(f'File: {entry.name}')
                elif entry.is_dir():
                    print(f'Directory: {entry.name}')

    def setup_reranker(self):
        self.reranker = T5ForConditionalGeneration.from_pretrained("t5-base", use_auth_token=os.getenv("HUGGINGFACE_API_KEY"))
        self.tokenizer = T5Tokenizer.from_pretrained("t5-base", use_auth_token=os.getenv("HUGGINGFACE_API_KEY"))
        print('Reranker setup complete')

    def rerank_documents(self, question, documents):
        inputs = [f"rank: {question} context: {doc['text']}" for doc in documents]
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.reranker.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=2)
        rankings = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        ranked_docs = sorted(zip(rankings, documents), key=lambda x: x[0])
        return [doc for rank, doc in ranked_docs]

    def initialize_chromadb(self):
        db_path = "testdemoAARON/chroma.db"
        client = db.PersistentClient(path=db_path)
        print(f'Collections: {client.list_collections()}')
        collection = client.get_collection(name="Company_Documents")
        return client, collection

    def setup_language_model(self):
        if self.llm_type == "External (OpenAI)" and self.api_key:
            try:
                self.repo_id = "openai/gpt-4o-mini"
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=self.api_key
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the external LLM: {e}")
        else:
            try:
                self.repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
                self.llm = HuggingFaceHub(
                    repo_id=self.repo_id,
                    model_kwargs={"temperature": 0.8, "top_p": 0.8, "top_k": 50},
                    huggingfacehub_api_token=os.getenv('HUGGINGFACE_API_KEY')
                )
            except Exception as e:
                raise ValueError(f"Failed to initialize the local LLM: {e}")

    def get_context_from_collection(self, input, access_levels):
        if len(access_levels) == 1:
            documents = self.collection.query(query_texts=[input], n_results=10, where=access_levels[0])
        else:
            documents = self.collection.query(query_texts=[input], n_results=10, where={"$or": access_levels})

        reranked_documents = self.rerank_documents(input, documents)
        context = " ".join([doc['text'] for doc in reranked_documents[:3]])
        return context

    def preprocess_input(self, input_dict):
        context = input_dict.get("context", "")
        question = input_dict.get("question", "")
        combined_text = f"{context} {question}"
        return combined_text

    def setup_langchain(self):
        template = """
        You are an informational chatbot. These employees will ask you questions about company data and meeting information. Use the following piece of context to answer the question.
        If you don't know the answer, just say you don't know. Please provide the file used for context.
        # You answer with short and concise answers, no longer than 2 sentences.

        Context: {context}
        Question: {question}
        Answer:
        """

        self.prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        self.rag_chain = (
            {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | AnswerOnlyOutputParser()
        )
