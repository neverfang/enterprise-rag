# Enterprise RAG - 企业级文档智能问答系统

> 基于 RAG 技术的企业级复杂文档智能问答系统，专注于解决海量异构文档的精准检索与结构化问答问题。

## 项目背景

在企业级文档处理场景中（如金融审计、投研分析），我们面临以下核心挑战：

1. **海量异构数据**：PDF 文档包含跨页表格、多栏排版、旋转报表等复杂结构
2. **跨文档推理**：约 30% 的问题需要对比多家公司的数据
3. **格式严格要求**：输出必须遵循严格的 JSON 格式，任何错误都会导致答案无效

本系统通过全链路工程优化，实现了对复杂文档的精准问答。

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Enterprise RAG 系统架构                    │
├─────────────────────────────────────────────────────────────┤
│  阶段一：文档解析与索引构建                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  PDF 解析    │ →  │  表格序列化  │ →  │  向量索引    │     │
│  │  (Docling)   │    │  (LLM驱动)  │    │  (FAISS)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  阶段二：智能检索与路由                                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  元数据路由  │ →  │  粗召回      │ →  │  LLM 精排   │     │
│  │  (精准定位)  │    │  (Top-30)    │    │  (重排序)    │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
├─────────────────────────────────────────────────────────────┤
│  阶段三：结构化生成与校验                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │  Prompt 构建 │ →  │  LLM 生成   │ →  │  自愈校验    │     │
│  │  (Pydantic)  │    │  (CoT推理)   │    │  (JSON修复)  │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

## 核心特性

### 1. 一文一库物理隔离架构
- 为每份文档构建独立的 FAISS 向量索引
- 通过元数据路由精准定位目标文档
- 完全避免跨文档信息干扰

### 2. 多级检索策略
- **粗召回**：基于 all-MiniLM-L6-v2 的向量检索，快速筛选 Top-30 候选
- **精排序**：LLM 驱动的语义重排序，提升上下文相关性
- **Parent Document Retrieval**：检索 chunk 粒度，返回完整页面作为上下文

### 3. 表格语义化处理
- LLM 驱动的表格序列化，将复杂表格转化为可检索的自然语言
- 解决财务表格难以被向量检索命中的问题

### 4. 自愈式生成框架
- **前置约束**：Pydantic Schema 严格定义输出格式
- **后置校验**：JSON Repair 自动修复语法错误
- **智能重试**：校验失败时将错误信息回传 LLM 触发重新生成

### 5. 多公司并行处理
- 自动识别问题中提及的多家公司
- 多线程并行检索与答案聚合
- 支持跨公司同口径数据对比分析

## 项目结构

```
enterprise-rag/
├── src/
│   ├── pdf_parsing.py          # PDF 深度解析
│   ├── tables_serialization.py # 表格语义化
│   ├── ingestion.py            # 数据摄入与索引构建
│   ├── retrieval.py            # 向量检索模块
│   ├── reranking.py            # LLM 重排序
│   ├── questions_processing.py # 问题处理与路由
│   ├── prompts.py              # Prompt 模板管理
│   ├── api_requests.py         # LLM API 调用与自愈
│   └── pipeline.py             # 流水线编排
├── data/
│   └── test_set/               # 测试数据集
├── main.py                     # CLI 入口
├── requirements.txt            # 依赖列表
└── README.md                   # 项目文档
```

## 快速开始

### 环境要求
- Python 3.10+
- GPU（推荐，用于 PDF 解析加速）

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/enterprise-rag.git
cd enterprise-rag

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\Activate.ps1  # Windows

# 安装依赖
pip install -e . -r requirements.txt

# 配置 API 密钥
cp env .env
# 编辑 .env 文件，填入你的 API 密钥
```

### 使用方法

```bash
# 查看可用命令
python main.py --help

# 1. 下载模型（首次使用）
python main.py download-models

# 2. 解析 PDF 文档
python main.py parse-pdfs --parallel

# 3. 构建向量索引
python main.py process-reports --config no_ser_tab

# 4. 运行问答
python main.py process-questions --config max_nst_o3m
```

## 配置说明

| 配置名 | 说明 |
|--------|------|
| `max_nst_o3m` | 使用 OpenAI o3-mini 的最佳性能配置 |
| `no_ser_tab` | 不使用序列化表格的通用配置 |
| `ser_tab` | 使用序列化表格的配置 |

## 技术栈

- **文档解析**：Docling + PyMuPDF
- **向量检索**：FAISS + all-MiniLM-L6-v2
- **重排序**：LLM-based Reranker
- **生成模型**：OpenAI GPT-4 / Gemini
- **结构化输出**：Pydantic Schema
- **自愈机制**：json_repair

## License

MIT
