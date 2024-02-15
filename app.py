import os
from langchain.document_loaders import OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone

class InferlessPythonModel:
    def initialize(self):
        #define the index name of Pinecone, embedding model name and pinecone API KEY
        index_name = "documents"
        embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        os.environ["PINECONE_API_KEY"] = "31b47ff0-5126-4f21-9d55-8ea2714e1a7d"

        #Initialize the embedding model, text_splitter & pinecone
        embeddings=HuggingFaceEmbeddings(model_name=embed_model_id)
        self.text_splitter=RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.pinecone = Pinecone(index_name=index_name, embedding=embeddings)

    def infer(self, inputs):
      pdf_link = inputs["pdf_url"]
      loader = OnlinePDFLoader(pdf_link)
      data = loader.load()
      documents = self.text_splitter.split_documents(data)
      response = self.pinecone.add_documents(documents)
      
      return {"result":response}

    def finalize(self):
      pass
