from llama_index.core import (
    VectorStoreIndex, 
    SimpleDirectoryReader, 
    ServiceContext,
    StorageContext,
    Settings
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.faiss import FaissVectorStore
import faiss

documents = SimpleDirectoryReader("./data").load_data()

embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

d = 384
faiss_index = faiss.IndexFlatL2(d)
faiss_store = FaissVectorStore(faiss_index=faiss_index)

storage_context = StorageContext.from_defaults(vector_store=faiss_store)

Settings.embed_model = embed_model

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
)

# faiss_store.save("faiss_index.index")
index.storage_context.persist("storage")