import logging

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis

LOG = logging.getLogger(__name__)

class VectorStores(object):
    def __init__(self, redis_url, index_name):
        self.redis_url = redis_url
        self.index_name = index_name
        self.model_name = 'sentence-transformers/all-MiniLM-L6-v2'

        self.embeddings = self._get_embeddings()

    def _get_embeddings(self):
        LOG.info(f"Loading encoding model {self.model_name}...")
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        model_kwargs = {'device': 'cpu'}
        embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

        return embeddings

    def store(self, splited_docs):
        LOG.info("Storing vector data to redis...")
        rds = Redis.from_documents(splited_docs, self.embeddings, redis_url=self.redis_url,  index_name=self.index_name)

        return rds

    def load(self):
        rds = Redis.from_existing_index(self.embeddings, redis_url=self.redis_url,  index_name=self.index_name)

        return rds
