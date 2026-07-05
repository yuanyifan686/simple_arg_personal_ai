from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

# ==================== 配置 ====================
KNOWLEDGE_DIR = "knowledge"           # 文档存放目录
PERSIST_DIR = "vector_db"             # 向量数据库存储目录
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文效果较好的本地模型
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
# =============================================

def create_vector_store():
    """加载文档 → 分块 → 向量化 → 存入 Chroma"""
    
    print("正在加载文档...")
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
        use_multithreading=True
    )
    documents = loader.load()
    print(f"共加载 {len(documents)} 个文档")

    print("正在进行文本分块...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "，", " "]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"共生成 {len(chunks)} 个文本块")

    print("正在生成向量并存入数据库...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    print("向量数据库构建完成！")
    return vectorstore


if __name__ == "__main__":
    create_vector_store()