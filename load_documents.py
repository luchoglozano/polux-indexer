import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# Configura tu clave de OpenAI si la usas en local o desde entorno
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Ruta al documento PDF que se va a indexar
pdf_path = "docs/tu-libro.pdf"

# Cargar PDF
loader = PyPDFLoader(pdf_path)
documents = loader.load()

# Dividir el texto en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

# Crear embeddings y guardar base vectorial
embedding = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory="./chroma_db"
)
vectordb.persist()

print("✅ Base vectorial generada exitosamente.")