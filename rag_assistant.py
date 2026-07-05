from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from rich.console import Console
from rich.markdown import Markdown
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ==================== 配置 ====================
PERSIST_DIR = "vector_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
MODEL_NAME = "MiniMax-M3"
BASE_URL = "https://api.minimax.chat/v1"
# =============================================

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )
    return vectorstore


def create_rag_chain(vectorstore):
    llm = ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=os.getenv("MINIMAX_API_KEY"),
        temperature=0.3,
    )

    # Prompt 模板
    template = """你是一个专业的知识库问答助手。
请严格根据下面提供的上下文内容回答问题。
如果上下文中没有相关信息，请诚实回答“根据现有资料无法回答”。

上下文：
{context}

问题：{question}
"""
    prompt = ChatPromptTemplate.from_template(template)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    # LCEL 链
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


def main():
    console.print("[bold green]✅ RAG 知识库问答系统已启动！[/bold green]")
    console.print("输入问题开始提问，输入 [bold]exit[/bold] 退出\n")

    vectorstore = load_vector_store()
    rag_chain = create_rag_chain(vectorstore)

    while True:
        try:
            question = console.input("[bold blue]你：[/bold blue]").strip()

            if question.lower() in ["exit", "quit", "退出"]:
                console.print("[yellow]再见！[/yellow]")
                break

            if not question:
                continue

            with console.status("[bold green]小助手思考中...[/bold green]"):
                answer = rag_chain.invoke(question)

            console.print("[bold green]小助手：[/bold green]")
            console.print(Markdown(answer))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]已退出[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]出错了：{str(e)}[/red]\n")


if __name__ == "__main__":
    main()