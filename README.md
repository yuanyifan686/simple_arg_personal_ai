# 混合智能助手 (Hybrid RAG Assistant)

一个支持**普通聊天 + 知识库问答**的命令行 AI 助手，具备本地 RAG 能力，可动态上传 PDF 文档，并显示答案来源。

## 项目简介

本项目将传统聊天机器人与 RAG（检索增强生成）技术结合，实现了一个既能正常对话，又能基于本地知识库精准回答问题的混合助手。所有数据本地存储，隐私性较好。

## 主要功能

- ✅ 普通多轮对话（支持记忆）
- ✅ 基于本地知识库的智能问答（RAG）
- ✅ 支持 TXT、MD、PDF 文件
- ✅ 动态上传 PDF（`/upload` 命令）
- ✅ 回答时显示参考来源（已去重）
- ✅ 本地向量化存储（Chroma + BGE 模型）
- ✅ 易于切换不同大模型（OpenAI 兼容接口）

## 技术栈

- Python + LangChain
- Chroma（本地向量数据库）
- HuggingFace Embeddings（BAAI/bge-small-zh-v1.5）
- OpenAI 兼容接口（支持 MiniMax、Grok 等）
- Rich（美化终端输出）

## 项目结构
rag-knowledge-assistant/
├── hybrid_assistant.py     # 主程序（混合助手）
├── pdf_loader.py           # PDF 加载与入库工具
├── requirements.txt        # 项目依赖清单
├── .env                    # 密钥、接口地址等环境配置
├── knowledge/              # 原始知识库文档目录
│   ├── company_policy.txt
│   └── company_policy.pdf
├── vector_db/              # Chroma 向量库持久化存储（程序自动生成）
└── README.md               # 项目说明文档
### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate          # Windows

pip install -r requirements.txt
2. 配置 API Key
创建 .env 文件：
envMINIMAX_API_KEY=sk-你的MiniMax_API_Key
3. 运行助手
Bashpython hybrid_assistant.py
使用说明
基本对话
直接输入问题即可，支持普通聊天和知识库问答。
上传 PDF
在对话中输入以下命令即可上传并加入知识库：
text/upload D:\path\to\your\file.pdf
上传成功后，后续提问可直接使用新内容。
查看来源
回答知识库相关问题时，底部会显示参考来源，例如：
text╭───────────────────────────────────── 📚 参考来源 ─────────────────────────────────────╮
│ • company_policy.pdf (第 1 页) │
│ • company_intro.txt │
╰───────────────────────────────────────────────────────────────────────────────────────╯
退出
输入 exit、quit 或 退出 即可退出程序。
配置说明

修改模型：在 hybrid_assistant.py 中调整 MODEL_NAME 和 BASE_URL
调整检索数量：修改 get_relevant_context_and_sources() 中的 k 值
记忆长度：修改 MAX_HISTORY 常量

注意事项

首次运行会自动下载 Embedding 模型（BAAI/bge-small-zh-v1.5），请保持网络通畅
建议所有文档统一使用 UTF-8 编码
扫描版 PDF 目前无法提取文字（需后续增加 OCR 支持）

未来优化方向

 支持更多文件格式（Word、Excel、网页）
 优化来源显示（显示片段内容）
 增加知识库管理命令（查看、删除文档）
 支持流式输出
 持久化对话历史
