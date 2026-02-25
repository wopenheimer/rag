import os
import sys

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Trocamos o loader por este aqui!
from langchain_community.document_loaders import PDFPlumberLoader
        
# ============================================================
# CONFIGURAÇÕES
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

COLLECTION_NAME = "meu_pdf"
PDF_PATH = "documento.pdf"

# ============================================================
# MODELOS
# ============================================================
llm = OllamaLLM(
    model="llama3",
    base_url=OLLAMA_BASE_URL,
    temperature=0.0 # Reduzimos a temperatura para ele ficar mais analítico e menos criativo
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url=OLLAMA_BASE_URL,
)

# ============================================================
# VECTOR STORE E INGESTÃO (AGORA SEM ESCONDER ERROS)
# ============================================================
print("🔗 Conectando ao banco de dados...")
vectorstore = PGVector(
    connection_string=DATABASE_URL,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Verifica se a coleção está vazia buscando qualquer coisa
resultados_existentes = vectorstore.similarity_search("teste_de_existencia", k=1)

if len(resultados_existentes) == 0:
    print("📄 Banco vazio. Iniciando processamento do PDF...")
    
    # PDFPlumber é nativo Python e respeita muito melhor a estrutura de tabelas
    loader = PDFPlumberLoader(PDF_PATH)
    
    print("📚 Carregando PDF...")
    documents = loader.load()
    print(f"📄 PDF lido com {len(documents)} páginas.")
    
    # Reduzi um pouco o chunk size. 4000 pode diluir muito a informação de uma tabela pequena.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, 
        chunk_overlap=400,
    )
    
    print("🔪 Criando chunks...")
    chunks = splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunks criados. Inserindo no banco (isso pode levar alguns segundos)...")
    
    # Se falhar aqui, o código vai "quebrar" e mostrar o erro real no terminal
    vectorstore.add_documents(chunks)
    print(f"✅ Inserção concluída com sucesso!")
else:
    print("📦 A tabela langchain_pg_embedding já possui dados. Pulando ingestão.")

# ============================================================
# RAG CHAIN E CHAT
# ============================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

prompt = ChatPromptTemplate.from_template("""
Você é um analista de dados preciso e direto. Responda a pergunta do usuário baseando-se EXCLUSIVAMENTE no contexto fornecido.

Regras de ouro:
1. Se a resposta exigir analisar uma tabela, verifique a qual coluna os números pertencem.
2. Se a resposta não estiver no contexto abaixo, diga "Não encontrei essa informação no documento". Não invente dados.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""")

def format_docs(docs):
    # Dica de dev: Se quiser depurar, descomente a linha abaixo para ver o que ele está achando:
    # print("\n--- CONTEXTO RECUPERADO ---\n", "\n\n".join(doc.page_content for doc in docs), "\n---------------------------\n")
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {
        "context": retriever | format_docs, 
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

print("\n🤖 Chatbot pronto! Digite 'sair' para encerrar.")
while True:
    pergunta = input("\nPergunta: ")

    if pergunta.lower() in ["sair", "exit"]:
        print("Até logo!")
        break
    
    resposta = rag_chain.invoke(pergunta)
    print("\nResposta:\n", resposta)