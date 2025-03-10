import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

class InferlessPythonModel:
    def initialize(self):
        #define the index name of Pinecone, embedding model name, LLM model name, and pinecone API KEY
        index_name = "documents"
        embed_model_id = "sentence-transformers/all-MiniLM-L6-v2"
        llm_model_id = "NousResearch/Nous-Hermes-llama-2-7b"
        snapshot_download(repo_id=embed_model_id,allow_patterns=["*.safetensors"])
        snapshot_download(repo_id=llm_model_id,allow_patterns=["*.safetensors"])
        os.environ["PINECONE_API_KEY"] = "YOUR_PINECONE_API_KEY"

        # Initialize the model for embeddings
        embeddings=HuggingFaceEmbeddings(model_name=embed_model_id)
        vectorstore = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)
        retriever = vectorstore.as_retriever()

        # Initialize the LLM
        tokenizer = AutoTokenizer.from_pretrained(llm_model_id)
        model = AutoModelForCausalLM.from_pretrained(llm_model_id,trust_remote_code=True,device_map="cuda")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=100)
        llm = HuggingFacePipeline(pipeline=pipe)

        # Define the chat template, and chain for retrival
        template = """Answer the question based only on the following context:
                      {context}
                      Question: {question}
                      """
        prompt = ChatPromptTemplate.from_template(template)
        self.chain = (
          RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
          | prompt
          | llm
          | StrOutputParser())
      
    def infer(self, inputs):
      question = inputs["question"]
      result = self.chain.invoke(question)
      text = result.split("Explanation:")[0][8:].split("\n")[0]
      return {"generated_result":result}

    def finalize(self):
      pass
