from llama_index.core import SimpleDirectoryReader

# documents = SimpleDirectoryReader(input_files=["./data/Resume020924RoderickC.pdf"]).load_data() 
documents = SimpleDirectoryReader("./data").load_data() 
print(documents)
print(f"\nTotal documents loaded: {len(documents)}")

# Load embedding model
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
embed_model = HuggingFaceEmbedding()

# Indexing and storing embedding to disk
import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext

db = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db.get_or_create_collection(name="my_collection")

vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context,
    embed_model=embed_model
)

# Step 4: Load embedding from disk
db2 = chromadb.PersistentClient(path="./chroma_db")
chroma_collection = db2.get_or_create_collection(name="my_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
index = VectorStoreIndex.from_vector_store(
    vector_store,
    embed_model=embed_model
)

# Step 5: Init Ollama and create LLM with Llama3.1
from llama_index.llms.ollama import Ollama # get local model with: ollama run <model_name>
llm = Ollama(model="gemma:2b", request_timeout=400)
print(llm)

# Step 6: Query


query_engine = index.as_query_engine(llm=llm)
# response = query_engine.query("What are the skills listed for this person?")
# print(response)

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