from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
import os
from dotenv import load_dotenv

load_dotenv()
console = Console()

# ==================== 配置 ====================
PERSIST_DIR = "vector_db"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"
MODEL_NAME = "MiniMax-M3"
BASE_URL = "https://api.minimax.chat/v1"

SYSTEM_PROMPT = """你是一个聪明、友好、专业的 AI 助手。
你既可以正常聊天，也可以基于知识库回答问题。
如果用户的问题和知识库相关，请优先使用知识库内容回答。
如果知识库中没有相关信息，请正常回答。
"""

MAX_HISTORY = 8
# =============================================

def load_vector_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return Chroma(persist_directory=PERSIST_DIR, embedding_function=embeddings)


def get_relevant_context_and_sources(vectorstore, question, k=3):
    """检索相关内容并返回来源"""
    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "", []

    context = "\n\n".join([doc.page_content for doc in docs])
    return context, docs


def create_llm():
    return ChatOpenAI(
        model=MODEL_NAME,
        base_url=BASE_URL,
        api_key=os.getenv("MINIMAX_API_KEY"),
        temperature=0.7,
    )


def format_sources(docs):
    """格式化来源并去重"""
    seen = set()
    sources = []
    
    for doc in docs:
        source = doc.metadata.get("source", "未知来源")
        page = doc.metadata.get("page", None)
        
        filename = os.path.basename(source)
        
        if page is not None:
            key = f"{filename}_page_{page}"
            display = f"• {filename} (第 {page + 1} 页)"
        else:
            key = filename
            display = f"• {filename}"
        
        if key not in seen:
            seen.add(key)
            sources.append(display)
    
    return "\n".join(sources) if sources else "未找到明确来源"


def chat():
    vectorstore = load_vector_store()
    llm = create_llm()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    console.print("[bold green]✅ 混合智能助手已启动！[/bold green]")
    console.print("支持普通聊天 + 知识库问答（带来源显示）")
    console.print("上传 PDF 命令：/upload 文件路径")
    console.print("输入 [bold]exit[/bold] 退出\n")

    while True:
        try:
            user_input = console.input("[bold blue]你：[/bold blue]").strip()

            if user_input.lower() in ["exit", "quit", "退出"]:
                console.print("[yellow]再见！[/yellow]")
                break

            # 处理上传 PDF 命令
            if user_input.lower().startswith("/upload "):
                pdf_path = user_input[8:].strip().strip('"')
                from pdf_loader import load_pdf_to_vectorstore
                load_pdf_to_vectorstore(pdf_path, PERSIST_DIR)
                continue

            if not user_input:
                continue

            # 检索知识库 + 获取来源
            context, source_docs = get_relevant_context_and_sources(vectorstore, user_input)

            # 构建 Prompt
            if context:
                prompt = f"""以下是知识库中的相关内容：
{context}

用户问题：{user_input}

请基于以上知识库内容（如果相关）或你的知识进行回答。
"""
            else:
                prompt = user_input

            messages.append({"role": "user", "content": prompt})

            with console.status("[bold green]思考中...[/bold green]"):
                response = llm.invoke(messages)

            answer = response.content
            messages.append({"role": "assistant", "content": answer})

            # 记忆管理
            if len(messages) > MAX_HISTORY + 1:
                messages = [messages[0]] + messages[-(MAX_HISTORY):]

            # 输出回答
            console.print("[bold green]小助手：[/bold green]")
            console.print(Markdown(answer))

            # 显示来源
            if source_docs:
                sources_text = format_sources(source_docs)
                console.print(Panel(sources_text, title="[bold cyan]📚 参考来源[/bold cyan]", border_style="cyan"))

            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]已退出[/yellow]")
            break
        except Exception as e:
            console.print(f"[red]出错了：{str(e)}[/red]\n")


if __name__ == "__main__":
    chat()