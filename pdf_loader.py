from langchain_community.document_loaders import PyPDFLoader, PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

def load_pdf_to_vectorstore(pdf_path: str, persist_dir: str = "vector_db"):
    """加载单个 PDF 并添加到向量数据库"""
    
    if not os.path.isfile(pdf_path):
        print(f"❌ 错误：文件不存在或不是文件 → {pdf_path}")
        return None
    
    if not pdf_path.lower().endswith(".pdf"):
        print("❌ 错误：只支持 PDF 文件")
        return None

    print(f"正在加载 PDF: {pdf_path}")

    # 优先使用 PDFPlumberLoader（中文支持更好）
    try:
        loader = PDFPlumberLoader(pdf_path)
        documents = loader.load()
    except Exception as e:
        print(f"PDFPlumber 加载失败，尝试使用 PyPDFLoader... ({e})")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

    print(f"✅ 成功提取 {len(documents)} 页内容")

    # 分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"生成 {len(chunks)} 个文本块")

    # 添加到已有向量数据库
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings
    )
    
    vectorstore.add_documents(chunks)
    print(f"✅ 成功将 PDF 内容添加到知识库！")

    return vectorstore


if __name__ == "__main__":
    print("=== PDF 知识库导入工具 ===")
    pdf_path = input("请输入 PDF 文件的完整路径: ").strip().strip('"')

    if pdf_path:
        load_pdf_to_vectorstore(pdf_path)
    else:
        print("未输入文件路径")