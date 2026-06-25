"""
pipeline.py - RAG流水线总调度中心

本文件定义了从PDF解析到最终问答的完整流水线，包含三个核心类：
- PipelineConfig: 路径配置管理
- RunConfig: 运行参数配置
- Pipeline: 流水线执行引擎

系统采用模块化设计，通过配置不同的参数组合，可以灵活切换各种RAG策略。
"""

from dataclasses import dataclass
from pathlib import Path
from pyprojroot import here
import logging
import os
import json
import pandas as pd

from src.pdf_parsing import PDFParser
from src.parsed_reports_merging import PageTextPreparation
from src.text_splitter import TextSplitter
from src.ingestion import VectorDBIngestor
from src.ingestion import BM25Ingestor
from src.questions_processing import QuestionsProcessor
from src.tables_serialization import TableSerializer


@dataclass
class PipelineConfig:
    """
    路径配置管理类 - 相当于项目的"目录地图"

    管理所有输入输出文件的路径，包括：
    - 原始数据（PDF、问题、元数据）
    - 中间产物（解析结果、合并报告、分块数据）
    - 最终输出（向量库、答案文件）

    关键设计：通过 serialized 参数区分是否使用序列化表格，
    路径会自动加上 _ser_tab 后缀，避免不同配置的数据互相覆盖。
    """
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", serialized: bool = False, config_suffix: str = ""):
        self.root_path = root_path
        # 根据是否使用序列化表格，动态生成路径后缀
        suffix = "_ser_tab" if serialized else ""

        # ========== 基础输入文件路径 ==========
        self.subset_path = root_path / subset_name           # 元数据文件（公司名、行业、货币等）
        self.questions_file_path = root_path / questions_file_name  # 问题文件
        self.pdf_reports_dir = root_path / pdf_reports_dir_name     # PDF报告目录

        # ========== 输出文件路径 ==========
        self.answers_file_path = root_path / f"answers{config_suffix}.json"  # 答案输出文件
        self.debug_data_path = root_path / "debug_data"                      # 调试数据目录
        self.databases_path = root_path / f"databases{suffix}"               # 数据库目录

        # ========== 数据库子目录 ==========
        self.vector_db_dir = self.databases_path / "vector_dbs"      # FAISS向量库（一文一库）
        self.documents_dir = self.databases_path / "chunked_reports" # 分块后的报告
        self.bm25_db_path = self.databases_path / "bm25_dbs"         # BM25关键词索引

        # ========== 中间产物目录（用于调试和检查） ==========
        self.parsed_reports_dirname = "01_parsed_reports"           # 第1步：解析后的JSON
        self.parsed_reports_debug_dirname = "01_parsed_reports_debug"  # Docling原始输出（非常详细）
        self.merged_reports_dirname = f"02_merged_reports{suffix}"  # 第2步：合并后的简化JSON
        self.reports_markdown_dirname = f"03_reports_markdown{suffix}" # 第3步：Markdown格式报告

        # 完整路径
        self.parsed_reports_path = self.debug_data_path / self.parsed_reports_dirname
        self.parsed_reports_debug_path = self.debug_data_path / self.parsed_reports_debug_dirname
        self.merged_reports_path = self.debug_data_path / self.merged_reports_dirname
        self.reports_markdown_path = self.debug_data_path / self.reports_markdown_dirname


@dataclass
class RunConfig:
    """
    运行配置类 - 流水线行为的"开关面板"

    通过调整这些参数，可以组合出从简单到复杂的各种RAG策略。
    推荐使用 max_nst_o3m 配置以获得最佳性能。
    """
    # ========== 表格处理 ==========
    use_serialized_tables: bool = False  # 是否使用序列化后的表格（Markdown格式）

    # ========== 检索策略 ==========
    parent_document_retrieval: bool = False  # 是否返回父页面而非小块（chunk），提升上下文完整性
    use_vector_dbs: bool = True              # 是否使用向量数据库（语义检索）
    use_bm25_db: bool = False                # 是否使用BM25（关键词检索）

    # ========== 重排序策略 ==========
    llm_reranking: bool = False              # 是否启用LLM重排序（用大模型对检索结果精排）
    llm_reranking_sample_size: int = 30      # 重排序时的候选数量（从向量库召回多少条）
    top_n_retrieval: int = 10                # 最终返回的文档数量（重排序后保留多少条）

    # ========== 并发控制 ==========
    parallel_requests: int = 10  # 并行请求线程数（同时处理多少个问题）

    # ========== 提交信息 ==========
    team_email: str = "79250515615@yandex.com"
    submission_name: str = "Ilia_Ris vDB + SO CoT"
    submission_file: bool = True
    pipeline_details: str = ""

    # ========== 模型配置 ==========
    full_context: bool = False               # 是否使用全文上下文（非RAG模式，把整篇文档塞给LLM）
    api_provider: str = "openai"             # API提供商：openai / local / gemini
    answering_model: str = "gpt-4o-mini-2024-07-18"  # 回答使用的模型
    config_suffix: str = ""                  # 配置后缀（用于区分不同配置的输出文件）


class Pipeline:
    """
    流水线执行引擎 - RAG系统的核心调度类

    提供从PDF解析到问答的完整方法链：
    1. parse_pdf_reports()      → 解析PDF为结构化JSON
    2. serialize_tables()       → 将表格序列化为Markdown（可选）
    3. merge_reports()          → 合并复杂JSON为简化结构
    4. export_reports_to_markdown() → 导出为Markdown（可选）
    5. chunk_reports()          → 将报告分块（用于向量化）
    6. create_vector_dbs()      → 创建FAISS向量库（一文一库）
    7. process_questions()      → 处理问题并生成答案

    使用方式：
    1. 实例化 Pipeline(root_path, run_config=某个配置)
    2. 按顺序调用上述方法
    3. 或直接调用 process_parsed_reports() 一键执行步骤3-6
    """
    def __init__(self, root_path: Path, subset_name: str = "subset.csv", questions_file_name: str = "questions.json", pdf_reports_dir_name: str = "pdf_reports", run_config: RunConfig = RunConfig()):
        self.run_config = run_config
        self.paths = self._initialize_paths(root_path, subset_name, questions_file_name, pdf_reports_dir_name)
        self._convert_json_to_csv_if_needed()

    def _initialize_paths(self, root_path: Path, subset_name: str, questions_file_name: str, pdf_reports_dir_name: str) -> PipelineConfig:
        """初始化路径配置，根据运行配置生成对应的目录结构"""
        return PipelineConfig(
            root_path=root_path,
            subset_name=subset_name,
            questions_file_name=questions_file_name,
            pdf_reports_dir_name=pdf_reports_dir_name,
            serialized=self.run_config.use_serialized_tables,
            config_suffix=self.run_config.config_suffix
        )

    def _convert_json_to_csv_if_needed(self):
        """
        兼容性处理：如果存在 subset.json 但不存在 subset.csv，
        则自动将 JSON 转换为 CSV 格式（后续流程统一使用CSV）
        """
        json_path = self.paths.root_path / "subset.json"
        csv_path = self.paths.root_path / "subset.csv"

        if json_path.exists() and not csv_path.exists():
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)

                df = pd.DataFrame(data)

                df.to_csv(csv_path, index=False)

            except Exception as e:
                print(f"Error converting JSON to CSV: {str(e)}")

# Docling在首次使用时会自动从HuggingFace下载模型
# 这里提前下载，避免运行时等待
    @staticmethod
    def download_docling_models():
        """下载Docling所需的模型（首次运行前调用）"""
        logging.basicConfig(level=logging.DEBUG)
        parser = PDFParser(output_dir=here())
        parser.parse_and_export(input_doc_paths=[here() / "src/dummy_report.pdf"])

    def parse_pdf_reports_sequential(self):
        """
        顺序解析PDF报告（单线程，适合调试）

        将PDF文件解析为结构化JSON，包含：
        - 文本内容
        - 表格数据
        - 元数据（页码等）
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        pdf_parser.parse_and_export(doc_dir=self.paths.pdf_reports_dir)
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def parse_pdf_reports_parallel(self, chunk_size: int = 2, max_workers: int = 10):
        """
        并行解析PDF报告（多进程，更快）

        Args:
            chunk_size: 每个worker处理的PDF数量
            max_workers: 并行worker进程数（建议设为CPU核心数）
        """
        logging.basicConfig(level=logging.DEBUG)

        pdf_parser = PDFParser(
            output_dir=self.paths.parsed_reports_path,
            csv_metadata_path=self.paths.subset_path
        )
        pdf_parser.debug_data_path = self.paths.parsed_reports_debug_path

        input_doc_paths = list(self.paths.pdf_reports_dir.glob("*.pdf"))

        pdf_parser.parse_and_export_parallel(
            input_doc_paths=input_doc_paths,
            optimal_workers=max_workers,
            chunk_size=chunk_size
        )
        print(f"PDF reports parsed and saved to {self.paths.parsed_reports_path}")

    def serialize_tables(self, max_workers: int = 10):
        """
        将表格序列化为Markdown格式（可选步骤）

        作用：将复杂的表格结构转换为LLM更容易理解的Markdown表格
        适用场景：当表格数据对问答很重要时启用
        """
        serializer = TableSerializer()
        serializer.process_directory_parallel(
            self.paths.parsed_reports_path,
            max_workers=max_workers
        )

    def merge_reports(self):
        """
        合并复杂JSON报告为简化结构

        输入：Docling解析的复杂嵌套JSON（包含文本块、表格、元数据）
        输出：按页组织的简化JSON，每页的文本块合并为一个字符串

        这一步是必要的，因为Docling的原始输出过于复杂，
        直接用于检索会导致上下文碎片化。
        """
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        _ = ptp.process_reports(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.merged_reports_path
        )
        print(f"Reports saved to {self.paths.merged_reports_path}")

    def export_reports_to_markdown(self):
        """
        导出为纯Markdown格式（用于调试或全文搜索模式）

        适用场景：
        - 人工检查解析结果是否正确
        - gemini_thinking 配置的全文上下文模式
        """
        ptp = PageTextPreparation(use_serialized_tables=self.run_config.use_serialized_tables)
        ptp.export_to_markdown(
            reports_dir=self.paths.parsed_reports_path,
            output_dir=self.paths.reports_markdown_path
        )
        print(f"Reports saved to {self.paths.reports_markdown_path}")

    def chunk_reports(self, include_serialized_tables: bool = False):
        """
        将报告分块（用于向量化）

        作用：将长文档切分为适合向量化的小块（chunk）
        分块策略在 TextSplitter 中定义
        """
        text_splitter = TextSplitter()

        serialized_tables_dir = None
        if include_serialized_tables:
            serialized_tables_dir = self.paths.parsed_reports_path

        text_splitter.split_all_reports(
            self.paths.merged_reports_path,
            self.paths.documents_dir,
            serialized_tables_dir
        )
        print(f"Chunked reports saved to {self.paths.documents_dir}")

    def create_vector_dbs(self):
        """
        创建FAISS向量数据库（核心步骤）

        关键设计："一文一库"物理隔离
        - 每份PDF生成独立的 .faiss 索引文件
        - 彻底杜绝不同公司数据的交叉干扰
        - 查询时通过元数据路由锁定目标公司的索引

        优势：检索精度高，完全可控的查询范围
        劣势：维护成百上千个小索引库，运维复杂度增加
        """
        input_dir = self.paths.documents_dir
        output_dir = self.paths.vector_db_dir

        vdb_ingestor = VectorDBIngestor()
        vdb_ingestor.process_reports(input_dir, output_dir)
        print(f"Vector databases created in {output_dir}")

    def create_bm25_db(self):
        """
        创建BM25关键词索引（可选）

        BM25是传统的关键词检索算法，与向量检索互补：
        - 向量检索：擅长语义相似性（"营收" ≈ "收入"）
        - BM25：擅长精确匹配（公司名、数字、专有名词）
        """
        input_dir = self.paths.documents_dir
        output_file = self.paths.bm25_db_path

        bm25_ingestor = BM25Ingestor()
        bm25_ingestor.process_reports(input_dir, output_file)
        print(f"BM25 database created at {output_file}")

    def parse_pdf_reports(self, parallel: bool = True, chunk_size: int = 2, max_workers: int = 10):
        """PDF解析入口：默认使用并行模式（更快）"""
        if parallel:
            self.parse_pdf_reports_parallel(chunk_size=chunk_size, max_workers=max_workers)
        else:
            self.parse_pdf_reports_sequential()

    def process_parsed_reports(self):
        """
        一键处理已解析的PDF报告（执行步骤2-5）

        流程：
        1. 合并报告（复杂JSON → 简化JSON）
        2. 导出Markdown（用于调试）
        3. 分块（为向量化做准备）
        4. 创建向量库（一文一库）

        注意：需要先调用 parse_pdf_reports() 完成步骤1
        """
        print("Starting reports processing pipeline...")

        print("Step 1: Merging reports...")
        self.merge_reports()

        print("Step 2: Exporting reports to markdown...")
        self.export_reports_to_markdown()

        print("Step 3: Chunking reports...")
        self.chunk_reports()

        print("Step 4: Creating vector databases...")
        self.create_vector_dbs()

        print("Reports processing pipeline completed successfully!")

    def _get_next_available_filename(self, base_path: Path) -> Path:
        """
        自动获取可用的文件名（避免覆盖已有结果）

        如果 answers.json 已存在，则返回 answers_01.json，
        如果 answers_01.json 也存在，则返回 answers_02.json，以此类推。
        """
        if not base_path.exists():
            return base_path

        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent

        counter = 1
        while True:
            new_filename = f"{stem}_{counter:02d}{suffix}"
            new_path = parent / new_filename

            if not new_path.exists():
                return new_path
            counter += 1

    def process_questions(self):
        """
        处理问题并生成答案（流水线的最终步骤）

        内部流程：
        1. 加载问题列表
        2. 对每个问题：
           a. 提取问题中的公司实体
           b. 路由到对应的向量库
           c. 检索相关文档
           d. （可选）LLM重排序
           e. 生成结构化答案
        3. 保存结果
        """
        processor = QuestionsProcessor(
            vector_db_dir=self.paths.vector_db_dir,
            documents_dir=self.paths.documents_dir,
            questions_file_path=self.paths.questions_file_path,
            use_metadata_routing=True,
            subset_path=self.paths.subset_path,
            parent_document_retrieval=self.run_config.parent_document_retrieval,
            llm_reranking=self.run_config.llm_reranking,
            llm_reranking_sample_size=self.run_config.llm_reranking_sample_size,
            top_n_retrieval=self.run_config.top_n_retrieval,
            parallel_requests=self.run_config.parallel_requests,
            api_provider=self.run_config.api_provider,
            answering_model=self.run_config.answering_model,
            full_context=self.run_config.full_context
        )

        output_path = self._get_next_available_filename(self.paths.answers_file_path)

        _ = processor.process_all_questions(
            output_path=output_path,
            submission_file=self.run_config.submission_file,
            team_email=self.run_config.team_email,
            submission_name=self.run_config.submission_name,
            pipeline_details=self.run_config.pipeline_details
        )
        print(f"Answers saved to {output_path}")


# =============================================================================
# 配置预设：从简单到复杂的演进路径
# =============================================================================

# 预处理配置：是否使用序列化表格
preprocess_configs = {"ser_tab": RunConfig(use_serialized_tables=True),      # 使用序列化表格
                      "no_ser_tab": RunConfig(use_serialized_tables=False)}  # 不使用序列化表格

# -----------------------------------------------------------------------------
# v0: 基础版 - 最简单的RAG配置
# -----------------------------------------------------------------------------
# 特点：向量检索 + 结构化输出 + 思维链推理
# 适用：快速验证流水线是否正常工作
base_config = RunConfig(
    parallel_requests=10,
    submission_name="Ilia Ris v.0",
    pipeline_details="Custom pdf parsing + vDB + Router + SO CoT; llm = GPT-4o-mini",
    config_suffix="_base"
)

# -----------------------------------------------------------------------------
# v1: 加入父文档检索
# -----------------------------------------------------------------------------
# 改进：返回整个父页面而非小块，提升上下文完整性
parent_document_retrieval_config = RunConfig(
    parent_document_retrieval=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.1",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_pdr"
)

# -----------------------------------------------------------------------------
# v2: 全功能版 - 加入表格序列化 + LLM重排序
# -----------------------------------------------------------------------------
# 改进：表格更易理解 + 检索结果更精准
max_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.2",
    pipeline_details="Custom pdf parsing + table serialization + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max"
)

# -----------------------------------------------------------------------------
# v3: 去掉表格序列化（测试其影响）
# -----------------------------------------------------------------------------
# 实验：验证表格序列化是否真的有帮助
max_no_ser_tab_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=20,
    submission_name="Ilia Ris v.3",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = GPT-4o",
    answering_model="gpt-4o-2024-08-06",
    config_suffix="_max_no_ser_tab"
)

# -----------------------------------------------------------------------------
# 配置1: 基础配置 - 使用 o3-mini 模型
# -----------------------------------------------------------------------------
# 核心配置：向量检索 + LLM重排序 + 父文档检索
max_nst_o3m_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    submission_name="Enterprise RAG v1",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m"
)

# -----------------------------------------------------------------------------
# 配置2: 表格序列化配置
# -----------------------------------------------------------------------------
# 在配置1基础上启用表格序列化
max_st_o3m_config = RunConfig(
    use_serialized_tables=True,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=25,
    submission_name="Enterprise RAG v2",
    pipeline_details="Custom pdf parsing + tables serialization + Router + vDB + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_st_o3m"
)

# -----------------------------------------------------------------------------
# 配置3: 使用本地部署的 Llama 70B 模型
# -----------------------------------------------------------------------------
# 适用于有本地 GPU 资源的场景
local_llama70b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    submission_name="Enterprise RAG v3",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; llm = llama-3.3-70b-instruct",
    api_provider="local",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_local_llama70b"
)

# -----------------------------------------------------------------------------
# 配置4: 轻量级 Llama 8B 模型
# -----------------------------------------------------------------------------
# 适用于资源受限的场景
local_llama8b_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=10,
    submission_name="Enterprise RAG v4",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT + SO reparser; llm = llama-3.1-8b-instruct",
    api_provider="local",
    answering_model="meta-llama/llama-3-1-8b-instruct",
    config_suffix="_local_llama8b"
)

# -----------------------------------------------------------------------------
# 配置5: Gemini 全文上下文模式
# -----------------------------------------------------------------------------
# 特点：利用 Gemini 超大上下文窗口，直接处理完整文档
gemini_thinking_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Enterprise RAG v5",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash-thinking-exp-01-21",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_fc"
)

# -----------------------------------------------------------------------------
# 配置6: Gemini Flash 全文上下文模式
# -----------------------------------------------------------------------------
gemini_flash_config = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=False,
    parallel_requests=1,
    full_context=True,
    submission_name="Enterprise RAG v6",
    pipeline_details="Custom pdf parsing + Full Context + Router + SO CoT + SO reparser; llm = gemini-2.0-flash",
    api_provider="gemini",
    answering_model="gemini-2.0-flash",
    config_suffix="_gemini_flash_fc"
)

# -----------------------------------------------------------------------------
# 配置7: 基础配置 + 更大上下文窗口
# -----------------------------------------------------------------------------
# 实验：增加检索数量（top_n=14）和重排序候选数（36）
max_nst_o3m_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.10",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = o3-mini; top_n = 14; topn for rerank = 36",
    answering_model="o3-mini-2025-01-31",
    config_suffix="_max_nst_o3m_bc"
)

# -----------------------------------------------------------------------------
# 配置8: 本地 Llama 70B + 更大上下文窗口
# -----------------------------------------------------------------------------
local_llama70b_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    llm_reranking=True,
    parallel_requests=5,
    llm_reranking_sample_size=36,
    top_n_retrieval=14,
    submission_name="Ilia Ris v.11",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + reranking + SO CoT; llm = llama-3.3-70b-instruct; top_n = 14; topn for rerank = 36",
    api_provider="local",
    answering_model="meta-llama/llama-3-3-70b-instruct",
    config_suffix="_local_llama70b_bc"
)

# -----------------------------------------------------------------------------
# v12: Gemini Thinking + 更大上下文窗口
# -----------------------------------------------------------------------------
gemini_thinking_config_big_context = RunConfig(
    use_serialized_tables=False,
    parent_document_retrieval=True,
    parallel_requests=1,
    top_n_retrieval=30,
    submission_name="Ilia Ris v.12",
    pipeline_details="Custom pdf parsing + vDB + Router + Parent Document Retrieval + SO CoT; llm = gemini-2.0-flash-thinking-exp-01-21; top_n = 30;",
    api_provider="gemini",
    answering_model="gemini-2.0-flash-thinking-exp-01-21",
    config_suffix="_gemini_thinking_bc"
)

# =============================================================================
# 配置字典：用于CLI命令行选择
# =============================================================================
configs = {"base": base_config,                           # 基础版
           "pdr": parent_document_retrieval_config,       # +父文档检索
           "max": max_config,                             # +表格序列化+LLM重排序
           "max_no_ser_tab": max_no_ser_tab_config,       # 去掉表格序列化
           "max_nst_o3m": max_nst_o3m_config,             # 推荐配置（最佳性能）
           "max_st_o3m": max_st_o3m_config,               # +表格序列化
           "local_llama70b": local_llama70b_config,       # 本地 Llama 70B
           "local_llama8b": local_llama8b_config,         # 本地 Llama 8B
           "gemini_thinking": gemini_thinking_config}     # Gemini全文模式


# =============================================================================
# 直接运行此文件的入口（用于调试）
# =============================================================================
# 使用方式：python .\src\pipeline.py
# 取消注释你想运行的方法即可
if __name__ == "__main__":
    root_path = here() / "data" / "test_set"
    pipeline = Pipeline(root_path, run_config=max_nst_o3m_config)


    # 第1步：将PDF解析为JSON（计算密集型，建议使用GPU）
    # 产出：debug_data/01_parsed_reports/ 目录下的JSON文件
    # pipeline.parse_pdf_reports_sequential()


    # 可选步骤：将表格序列化为Markdown格式
    # 仅在使用 use_serialized_tables=True 的配置时需要
    # pipeline.serialize_tables(max_workers=5)


    # 第2步：合并复杂JSON为简化结构
    # 产出：debug_data/02_merged_reports/ 目录下的JSON文件
    # pipeline.merge_reports()


    # 可选步骤：导出为纯Markdown格式
    # 用于调试检查或 gemini_thinking 全文模式
    # 产出：debug_data/03_reports_markdown/ 目录下的Markdown文件
    # pipeline.export_reports_to_markdown()


    # 第3步：将报告分块（为向量化做准备）
    # 产出：databases/chunked_reports/ 目录下的JSON文件
    # pipeline.chunk_reports()


    # 第4步：创建FAISS向量库（一文一库）
    # 产出：databases/vector_dbs/ 目录下的.faiss文件
    # pipeline.create_vector_dbs()


    # 第5步：处理问题并生成答案
    # 产出：answers.json（答案文件）
    # pipeline.process_questions()
