from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.llms.ollama import Ollama
import chromadb

# Step 1: Load embedding model (must match one used in build step)
embed_model = HuggingFaceEmbedding()

# Step 2: Load vector store from disk
db = chromadb.PersistentClient(path="./vectors/chroma_db_indexed-smallerchunk")
chroma_collection = db.get_or_create_collection(name="my_collection-smallerchunk")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

# Step 3: Load index from vector store
index = VectorStoreIndex.from_vector_store(
    vector_store=vector_store,
    embed_model=embed_model
)

# Step 4: Load LLM
llm = Ollama(model="gemma:2b", request_timeout=400)

# Step 5: Create query engine and run chat
query_engine = index.as_query_engine(
    llm=llm, 
    similarity_top_k=4
)

print("\nType your question (or type 'exit' to quit):")
while True:
    user_input = input(">>> ")
    if user_input.lower() in ["exit", "quit"]:
        break
    try:
        response = query_engine.query(user_input)
        print("\n" + str(response) + "\n")
    except Exception as e:
        print(f"Error: {e}")
