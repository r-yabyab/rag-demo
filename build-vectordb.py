from config import VECTOR_DB_PATH, COLLECTION_NAME
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
import chromadb

# Step 1: Load documents
documents = SimpleDirectoryReader("./data/apa-papers").load_data()
print(f"Total documents loaded: {len(documents)}")

# Step 2: Load embedding model
embed_model = HuggingFaceEmbedding()

Settings.chunk_size = 512 
Settings.chunk_overlap = 50

# Step 3: Set up persistent ChromaDB vector store
db = chromadb.PersistentClient(path=VECTOR_DB_PATH)
chroma_collection = db.get_or_create_collection(name=COLLECTION_NAME)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Step 4: Build and persist index
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

print("vectordb created")
