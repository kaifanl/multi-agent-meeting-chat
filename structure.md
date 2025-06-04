graph LR
    subgraph "用户交互层 (User Interface)"
        UI_Input["用户输入: 音频/文本/问题"] --> Coord;
        Coord --> UI_Output["用户输出: 会议纪要/答案"];
    end

    subgraph "多智能体系统 (Multi-Agent System)"
        Coord["A1: 协调与接口智能体<br>(Coordinator & Interface Agent)"]

        subgraph "会议纪要生成流程"
            Coord -- "发起转录任务" --> Trans;
            Trans["A2: 语音转录智能体<br>(Transcription Agent)"] -- "转录文本" --> AE;
            AE["A3: 内容分析与提取智能体<br>(Analysis & Extraction Agent)"] -- "结构化信息" --> Format;
            Format["A4: 纪要生成与格式化智能体<br>(Minutes Generation & Formatting Agent)"] -- "格式化纪要" --> Coord;
            Format -- "格式化纪要 (用于RAG)" --> RAG_Index;
        end

        subgraph "RAG问答流程"
            Coord -- "发起问答任务 (用户问题)" --> RAG_Query;
            RAG_Index["知识库构建模块<br>(纪要切分与向量化)"];
            RAG_Query["A5: RAG问答智能体<br>(RAG Q&A Agent - 查询处理)"];
            RAG_Query -- "检索请求" --> RAG_Store["向量数据库/索引<br>(Vector Store/Index)"];
            RAG_Store -- "相关文本块" --> RAG_Query;
            RAG_Query -- "上下文+问题" --> LLM_QA["LLM (问答生成)"];
            LLM_QA -- "生成的答案" --> Coord;
        end

        subgraph "外部服务 (External Services)"
            Trans -- "音频数据" --> STT_API["语音转文本API<br>(e.g., OpenAI Whisper)"];
            STT_API -- "文本" --> Trans;

            AE -- "文本+提取Prompt" --> LLM_Extract["LLM (信息提取)"];
            LLM_Extract -- "结构化信息" --> AE;

            RAG_Index -- "文本块" --> Embedding_API["文本嵌入API<br>(e.g., OpenAI Embeddings)"];
            Embedding_API -- "向量" --> RAG_Index;

            RAG_Query -- "文本块(用于生成Prompt)" --> Embedding_API;
            RAG_Query -- "用户问题" --> Embedding_API;

        end
    end

    style UI_Input fill:#lightgrey,stroke:#333,stroke-width:2px
    style UI_Output fill:#lightgrey,stroke:#333,stroke-width:2px
    style Coord fill:#lightblue,stroke:#333,stroke-width:2px
    style Trans fill:#lightgreen,stroke:#333,stroke-width:2px
    style AE fill:#lightgreen,stroke:#333,stroke-width:2px
    style Format fill:#lightgreen,stroke:#333,stroke-width:2px
    style RAG_Index fill:#lightyellow,stroke:#333,stroke-width:2px
    style RAG_Query fill:#lightyellow,stroke:#333,stroke-width:2px
    style RAG_Store fill:#orange,stroke:#333,stroke-width:2px
    style STT_API fill:#f9f,stroke:#333,stroke-width:2px
    style LLM_Extract fill:#f9f,stroke:#333,stroke-width:2px
    style LLM_QA fill:#f9f,stroke:#333,stroke-width:2px
    style Embedding_API fill:#f9f,stroke:#333,stroke-width:2px

    %% 交互流程箭头说明
    %% 用户输入 -> 协调器
    %% 协调器 -> 转录 -> 分析提取 -> 格式化 -> 协调器 (纪要)
    %% 格式化 -> RAG索引构建