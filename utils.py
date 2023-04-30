import logging

from transformers import pipeline
from langchain.llms import HuggingFacePipeline

from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

LOG = logging.getLogger(__name__)

def documents_loader(files, chunk_size, chunk_overlap):
    documents = []
    for name in files:
        suffix = name.split(".")[-1]
        if suffix == "txt":
            loader = TextLoader(name)
        elif suffix == "pdf":
            loader = UnstructuredPDFLoader(name)
        else:
            LOG.warning(f"Currently document {name} is not supported")

        documents += loader.load()

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len)

    splited_docs = text_splitter.split_documents(documents)

    return splited_docs

def make_pipeline(model, tokenizer):
    LOG.info("creating transformer pipeline...")
    model_pipeline = pipeline("text-generation",
                               model=model,
                               tokenizer=tokenizer,
                               device=0,
                               max_length=512)

    return model_pipeline

# you can specify RetrievalQA and use it to fetch docs along with answer
def make_chain(pipeline):
    LOG.info("creating chain...")
    llm = HuggingFacePipeline(pipeline=pipeline)

    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain_type = "stuff"
    # chain_type = "map_rerank"
    LOG.info("Loading Q&A chain...")
    chain = load_qa_chain(llm, chain_type=chain_type, prompt=prompt)

    # debug the prompt
    LOG.debug(chain.llm_chain.prompt.template)

    return chain
