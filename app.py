import os
import sys
import pdfplumber
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ============================================================
# CONFIGURAÇÕES
# ============================================================
DATABASE_URL = os.getenv("DATABASE_URL")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

COLLECTION_NAME = "edital_ifsuldeminas_v2" # Mudamos o nome para recriar com a nova lógica
PDF_PATH = "documento.pdf"

# ============================================================
# FUNÇÃO DE EXTRAÇÃO MELHORADA (PDF -> TEXTO + MARKDOWN)
# ============================================================
def carregar_pdf_com_tabelas(pdf_path):
    print(f"📂 Abrindo PDF: {pdf_path} com PDFPlumber...")
    documents = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            # 1. Extrai o texto normal
            text = page.extract_text() or ""
            
            # 2. Extrai as tabelas de forma estruturada
            tables = page.extract_tables()
            table_markdown = ""
            
            if tables:
                for table in tables:
                    for row in table:
                        # Limpa quebras de linha dentro das células para não quebrar o Markdown
                        cells = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                        table_markdown += "| " + " | ".join(cells) + " |\n"
                    table_markdown += "\n" # Espaço entre tabelas
            
            # 3. Monta o conteúdo final da página
            content = f"--- PÁGINA {i+1} ---\n{text}\n\n### TABELAS E DADOS ESTRUTURADOS:\n{table_markdown}"
            
            # Criamos o objeto Document do LangChain
            doc = Document(
                page_content=content,
                metadata={"page": i+1, "source": pdf_path}
            )
            documents.append(doc)
            
    return documents

# ============================================================
# MODELOS
# ============================================================
llm = OllamaLLM(
    model="llama3",
    base_url=OLLAMA_BASE_URL,
    temperature=0.0
)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text", 
    base_url=OLLAMA_BASE_URL,
)

# ============================================================
# VECTOR STORE E INGESTÃO
# ============================================================
vectorstore = PGVector(
    connection_string=DATABASE_URL,
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)

# Verifica se precisa ingerir
resultados_existentes = vectorstore.similarity_search("teste", k=1)

if len(resultados_existentes) == 0:
    print("🚀 Processando edital com nova estratégia de tabelas...")
    
    # Usa a nova função em vez do loader padrão
    documents = carregar_pdf_com_tabelas(PDF_PATH)
    print(f"✅ {len(documents)} páginas processadas.")
    
    # Splitter ajustado para não quebrar linhas de tabelas Markdown
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, 
        chunk_overlap=400,
        separators=["\n--- PÁGINA", "\n### TABELAS", "\n\n", "\n", " "]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ {len(chunks)} chunks criados. Inserindo no PGVector...")
    vectorstore.add_documents(chunks)
else:
    print("📦 Banco de dados já populado.")

# ============================================================
# RAG CHAIN
# ============================================================
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Prompt refinado para ser mais rigoroso com os nomes dos Campi e cursos
prompt = ChatPromptTemplate.from_template("""
Você é um assistente especializado no Edital do IFSULDEMINAS. 
Responda de forma objetiva usando APENAS o contexto abaixo.

REGRAS CRÍTICAS:
1. Ao citar vagas, confirme se o CAMPUS e o CURSO correspondem exatamente à pergunta.
2. As tabelas estão em formato Markdown. Leia as colunas com atenção para não trocar os valores das cotas.
3. Se a informação não estiver clara ou estiver ausente, diga explicitamente que não encontrou.

Contexto:
{context}

Pergunta:
{question}

Resposta:
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

# ============================================================
# CHAT LOOP
# ============================================================
print("\n🤖 Chatbot do Edital Pronto!")
while True:
    pergunta = input("\nPergunta (ex: 'Quantas vagas para Agronomia em Inconfidentes?'): ")
    if pergunta.lower() in ["sair", "exit"]: break
    
    print("\nAnalisando...")
    print(rag_chain.invoke(pergunta))