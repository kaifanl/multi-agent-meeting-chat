import streamlit as st
import time
import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Ensure your agent files are in the correct location
from analysis_agent import analyze_meeting_text_v2
# from transcript_agent import transcript_meeting_text # Using dummy below
from qwen_api import APILLMResponse # Assuming this is your LLM API wrapper

# --- RAG Configuration ---
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Characters overlap between chunks
EMBEDDING_DIM = 384  # Example dimension, replace if using a real model with different dim
TOP_K_CHUNKS = 5   # Number of top relevant chunks to retrieve for context

# --- Dummy Agent Functions & RAG Helpers (If not using real agents/models) ---
def transcript_meeting_text(audio_file_path):
    """Dummy function for transcript_meeting_text."""
    st.info(f"DUMMY TRANSCRIPT: Simulating transcription for {os.path.basename(audio_file_path)}...")
    time.sleep(5) # Simulate processing time
    if 'huggingcast' in audio_file_path.lower():
        # Ensure this path is correct and the file exists for the dummy to work
        try:
            return open('/home/dell/mwy/AIML-project/huggingcast-s2e6 text-en.txt', 'r', encoding='utf-8').read()
        except FileNotFoundError:
            st.error("Dummy transcript file (huggingcast) not found. Returning placeholder.")
            return "This is a placeholder dummy transcript for huggingcast because the file was not found."
    elif 'sora' in audio_file_path.lower():
        try:
            return open('/home/dell/mwy/AIML-project/sora.txt', 'r', encoding='utf-8').read()
        except FileNotFoundError:
            st.error("Dummy transcript file (sora) not found. Returning placeholder.")
            return "This is a placeholder dummy transcript for sora because the file was not found."
    else:
        return f"这是从 {os.path.basename(audio_file_path)} 转录的通用模拟文本。会议富有成效。"

# def analyze_meeting_text_v2(transcript_text, meeting_name="会议"):
# """Dummy function for analyze_meeting_text_v2."""
#     st.info(f"DUMMY SUMMARY: Analyzing transcript for {meeting_name}...")
#     time.sleep(2)
#     return f"""
# 会议名称: {meeting_name}
# (基于模拟转录稿生成的摘要)
# 快速概览:
# -   要点1 from transcript.
# -   要点2 from transcript.
# """

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Splits text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start + chunk_overlap >= len(text) and end < len(text): # Ensure last part is captured
            chunks.append(text[len(text)-(chunk_size-chunk_overlap):]) # Adjust if last chunk is too small
            break
    # A simple way to ensure the very last piece of text is included if the loop logic misses it
    if not chunks or len(text) > (len(chunks) * (chunk_size - chunk_overlap) - chunk_overlap):
        if len(text) > chunk_size:
            final_chunk_start = max(0, len(text) - chunk_size)
            if not chunks or chunks[-1] != text[final_chunk_start:]:
                 chunks.append(text[final_chunk_start:])
        elif text and (not chunks or chunks[-1] != text):
            chunks.append(text)
    # Filter out very short or empty chunks that might result from edge cases
    return [c for c in chunks if len(c.strip()) > 10]


# !!! IMPORTANT: Replace this with a real embedding model !!!
# For example, using sentence-transformers:
# from sentence_transformers import SentenceTransformer
# embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Load once globally
# EMBEDDING_DIM = embedding_model.get_sentence_embedding_dimension()
# def get_embedding(text_chunk):
#     if not text_chunk.strip():
#         return [0.0] * EMBEDDING_DIM
#     return embedding_model.encode(text_chunk).tolist()

def get_embedding(text_chunk):
    """Dummy function to generate random embeddings."""
    # st.info(f"DUMMY EMBEDDING: Generating for chunk: '{text_chunk[:30]}...'")
    if not text_chunk.strip():
        return [0.0] * EMBEDDING_DIM
    # np.random.seed(len(text_chunk)) # Make dummy embeddings somewhat deterministic for a given chunk for stability
    return np.random.rand(EMBEDDING_DIM).tolist()
# --- End of Dummy Agent Functions & RAG Helpers ---


# --- Configuration ---
st.set_page_config(layout="wide", page_title="会议纪要与聊天系统")

# --- Session State Initialization ---
if 'uploaded_meetings' not in st.session_state:
    # Structure: {meeting_name: {status: str, original_file_name: str, audio_file_path: str,
    #                             transcript_file_path: str, summary_file_path: str,
    #                             rag_data_file_path: str, # <-- NEW for RAG
    #                             safe_filename_base: str, transcript_content_in_memory: str,
    #                             summary_content: str}}
    st.session_state.uploaded_meetings = {}
if 'selected_meetings_for_chat' not in st.session_state:
    st.session_state.selected_meetings_for_chat = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = {}
if 'show_upload_form' not in st.session_state:
    st.session_state.show_upload_form = False
if 'processing_queue' not in st.session_state:
    st.session_state.processing_queue = []
if 'show_summary_for_meeting_name' not in st.session_state:
    st.session_state.show_summary_for_meeting_name = None

# --- Directory Setup ---
AUDIO_DIR = "audio"
SUMMARY_DIR = "summary"
TRANSCRIPT_DIR = "transcript"
RAG_DATA_DIR = "rag_data" # <-- NEW DIRECTORY for RAG JSON files
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(SUMMARY_DIR, exist_ok=True)
os.makedirs(TRANSCRIPT_DIR, exist_ok=True)
os.makedirs(RAG_DATA_DIR, exist_ok=True) # <-- CREATE IT

# --- Backend Processing Function ---
def process_meeting_file(meeting_name, uploaded_file_obj):
    safe_filename_base = "".join(c if c.isalnum() or c in (' ', '_', '-') else '_' for c in meeting_name).replace(' ', '_')

    audio_filename = f"{safe_filename_base}.m4a"
    audio_file_path = os.path.join(AUDIO_DIR, audio_filename)

    transcript_filename = f"{safe_filename_base}_transcript.txt"
    transcript_file_path = os.path.join(TRANSCRIPT_DIR, transcript_filename)

    summary_filename = f"{safe_filename_base}_summary.txt"
    summary_file_path = os.path.join(SUMMARY_DIR, summary_filename)

    rag_data_filename = f"{safe_filename_base}_rag.json" # <-- NEW
    rag_data_file_path = os.path.join(RAG_DATA_DIR, rag_data_filename) # <-- NEW

    st.session_state.uploaded_meetings[meeting_name]['audio_file_path'] = audio_file_path
    st.session_state.uploaded_meetings[meeting_name]['transcript_file_path'] = transcript_file_path
    st.session_state.uploaded_meetings[meeting_name]['summary_file_path'] = summary_file_path
    st.session_state.uploaded_meetings[meeting_name]['rag_data_file_path'] = rag_data_file_path # <-- STORE PATH
    st.session_state.uploaded_meetings[meeting_name]['safe_filename_base'] = safe_filename_base

    processing_steps = [
        ("保存音频", None),
        ("转录音频", transcript_meeting_text),
        ("保存转录稿", None),
        ("文本分块与向量化", None), # <-- NEW RAG STEP
        ("保存RAG数据", None),     # <-- NEW RAG STEP
        ("生成摘要", analyze_meeting_text_v2),
        ("保存摘要", None)
    ]
    total_steps = len(processing_steps)
    current_step_num = 0

    try:
        current_step_num = 1 # 保存音频
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        with open(audio_file_path, "wb") as f:
            f.write(uploaded_file_obj.getvalue())
        time.sleep(0.1)

        current_step_num = 2 # 转录音频
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        st.info(f"Transcribing {os.path.basename(audio_file_path)}...")
        transcript_text = transcript_meeting_text(audio_file_path)
        if not transcript_text or not isinstance(transcript_text, str):
            raise ValueError("转录结果为空或格式不正确。")
        st.session_state.uploaded_meetings[meeting_name]['transcript_content_in_memory'] = transcript_text
        time.sleep(0.1)

        current_step_num = 3 # 保存转录稿
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        with open(transcript_file_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)
        time.sleep(0.1)

        current_step_num = 4 # 文本分块与向量化
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        st.info(f"Chunking and embedding transcript for {meeting_name}...")
        text_chunks = chunk_text(transcript_text)
        rag_data_list = []
        for i, chunk in enumerate(text_chunks):
            embedding = get_embedding(chunk) # Uses dummy embedding
            rag_data_list.append({
                "chunk_id": f"chunk_{i}",
                "meeting_id": meeting_name, # Using the display name as meeting_id
                "text": chunk,
                "embedding": embedding
            })
        if not rag_data_list:
            st.warning(f"No RAG data generated for {meeting_name}, possibly empty or very short transcript.")
        time.sleep(0.1)

        current_step_num = 5 # 保存RAG数据
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        with open(rag_data_file_path, "w", encoding="utf-8") as f:
            json.dump(rag_data_list, f, ensure_ascii=False, indent=2)
        time.sleep(0.1)

        current_step_num = 6 # 生成摘要
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        st.info(f"Summarizing {os.path.basename(transcript_file_path)}...")
        summary_content = analyze_meeting_text_v2(transcript_text, meeting_name=meeting_name) # Pass meeting_name
        if not summary_content or not isinstance(summary_content, str):
            raise ValueError("摘要结果为空或格式不正确。")
        time.sleep(0.1)

        current_step_num = 7 # 保存摘要
        st.session_state.uploaded_meetings[meeting_name]['status'] = f"处理中: {processing_steps[current_step_num-1][0]} ({current_step_num}/{total_steps})"
        with open(summary_file_path, "w", encoding="utf-8") as f:
            f.write(summary_content)
        st.session_state.uploaded_meetings[meeting_name]['summary_content'] = summary_content
        time.sleep(0.1)

        st.session_state.uploaded_meetings[meeting_name]['status'] = 'processed'
        st.success(f"会议 “{meeting_name}” 处理完成。")

    except Exception as e:
        error_step_name = "未知步骤"
        if 0 < current_step_num <= len(processing_steps):
            error_step_name = processing_steps[current_step_num-1][0]
        st.error(f"处理会议 “{meeting_name}” 时在步骤 '{error_step_name}' 发生错误: {e}")
        st.session_state.uploaded_meetings[meeting_name]['status'] = f'错误: {error_step_name}失败'
    finally:
        st.rerun()


# --- RAG Chat API Function ---
def get_chat_response_rag(user_query, selected_meeting_names):
    """
    Gets a chat response using RAG based on selected meetings and a user query.
    """
    llm_client = APILLMResponse()

    st.info(f"RAG: Embedding user query: '{user_query[:50]}...'")
    query_embedding = np.array(get_embedding(user_query)).reshape(1, -1) # Reshape for cosine_similarity

    all_chunks_data = []
    loaded_rag_meeting_names = []

    for meeting_name in selected_meeting_names:
        meeting_data = st.session_state.uploaded_meetings.get(meeting_name)
        if meeting_data and 'rag_data_file_path' in meeting_data:
            rag_path = meeting_data['rag_data_file_path']
            if os.path.exists(rag_path):
                try:
                    with open(rag_path, "r", encoding="utf-8") as f:
                        meeting_rag_data = json.load(f)
                        all_chunks_data.extend(meeting_rag_data) # Add meeting_id to each chunk if not already there
                        loaded_rag_meeting_names.append(meeting_name)
                        st.info(f"RAG: Loaded {len(meeting_rag_data)} chunks for {meeting_name}")
                except Exception as e:
                    st.warning(f"RAG:无法读取会议 {meeting_name} 的RAG数据: {e}")
            else:
                st.warning(f"RAG:找不到会议 {meeting_name} 的RAG数据文件: {rag_path}")
        else:
            st.warning(f"RAG:会议 {meeting_name} 没有可用的RAG数据路径。")

    if not all_chunks_data:
        return "错误：未能加载任何所选会议的RAG数据（文本块和向量）。无法进行对话。"

    # Perform similarity search
    chunk_embeddings = np.array([chunk['embedding'] for chunk in all_chunks_data])
    if chunk_embeddings.ndim == 1: # Handle case of single chunk with 1D embedding array
        if chunk_embeddings.shape[0] == query_embedding.shape[1]: # Check if it's a single embedding vector
             chunk_embeddings = chunk_embeddings.reshape(1, -1)
        else:
            return "错误: RAG数据中的向量维度不一致或格式错误。"


    if query_embedding.shape[1] != chunk_embeddings.shape[1]:
        st.error(f"Query embedding dim ({query_embedding.shape[1]}) != Chunk embedding dim ({chunk_embeddings.shape[1]})")
        return "错误：查询向量与文本块向量的维度不匹配。请检查Embedding模型配置。"

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    
    # Get top-K relevant chunks
    # Ensure there are enough similarities to sort; handle cases with fewer chunks than TOP_K_CHUNKS
    num_chunks_to_consider = min(len(similarities), TOP_K_CHUNKS * 2) # Get more to avoid empty context if some are bad
    
    # Ensure we have at least one similarity score before trying to get indices
    if len(similarities) == 0:
        return "错误: 没有可比较的文本块来回答您的问题。"

    # Get indices of top-K similarities
    # If TOP_K_CHUNKS is larger than available similarities, take all available
    k_to_retrieve = min(TOP_K_CHUNKS, len(similarities))
    if k_to_retrieve == 0: # Should be caught by len(similarities) == 0 above, but defensive.
        return "错误: 没有足够的文本块可以检索来回答您的问题。"
        
    top_k_indices = np.argsort(similarities)[-k_to_retrieve:][::-1]


    relevant_chunks_text = []
    retrieved_chunk_info = [] # For debugging or more detailed display
    for i in top_k_indices:
        chunk = all_chunks_data[i]
        relevant_chunks_text.append(f"--- 来自会议: {chunk['meeting_id']} (相似度: {similarities[i]:.4f}) ---\n{chunk['text']}\n--- 结束片段 ---")
        retrieved_chunk_info.append({
            "meeting": chunk['meeting_id'],
            "chunk_id": chunk['chunk_id'],
            "similarity": float(similarities[i]),
            "text_preview": chunk['text'][:100] + "..."
        })
    
    # st.write("DEBUG: Retrieved Chunks Info:", retrieved_chunk_info) # For debugging

    if not relevant_chunks_text:
        return "未能从所选会议中找到与您问题相关的具体信息。请尝试换个问法或检查会议内容。"

    rag_context = "\n\n".join(relevant_chunks_text)

    # Construct messages for LLM
    if len(loaded_rag_meeting_names) > 1:
        meetings_context_intro = f"以下是根据您的问题从会议 {', '.join(loaded_rag_meeting_names)} 中检索到的相关文本片段。"
    else:
        meetings_context_intro = f"以下是根据您的问题从会议 {loaded_rag_meeting_names[0]} 中检索到的相关文本片段。"

    messages = [
        {'role': 'system', 'content': f"""
         您是一个乐于助人的AI助手。
         - 您的任务是根据提供的会议记录相关片段来回答用户的问题。
         - {meetings_context_intro}
         - 请确保您的回答严格基于这些提供的文本片段。
         - 如果信息未在片段中出现，请明确指出。
         - 您的语言应与用户提问的语言保持一致。
         """},
        {'role': 'user', 'content': f"""
         会议相关文本片段:
         {rag_context}

         用户的问题:
         {user_query}
         """}
    ]
    # st.write("DEBUG: Sending to LLM (RAG):", messages) # For debugging
    return llm_client.get_response(messages)


# --- UI Rendering ---

# --- Sidebar ---
with st.sidebar:
    if st.button("➕ 新的会议", use_container_width=True, type="secondary"):
        st.session_state.show_upload_form = True
        st.session_state.show_summary_for_meeting_name = None

    if st.session_state.show_upload_form:
        with st.expander("上传新会议文件", expanded=True):
            with st.form("new_meeting_form", clear_on_submit=True):
                new_meeting_name = st.text_input("会议名称", placeholder="例如：2024年Q3战略规划会")
                uploaded_m4a_file = st.file_uploader("选择 .m4a 文件", type=['m4a'])
                submit_button = st.form_submit_button("开始处理")

                if submit_button:
                    if not new_meeting_name:
                        st.warning("请输入会议名称。")
                    elif not uploaded_m4a_file:
                        st.warning("请上传 .m4a 文件。")
                    elif new_meeting_name in st.session_state.uploaded_meetings:
                        st.warning(f"会议名称 “{new_meeting_name}” 已存在。")
                    else:
                        st.session_state.uploaded_meetings[new_meeting_name] = {
                            'status': 'queued',
                            'original_file_name': uploaded_m4a_file.name,
                        }
                        st.session_state.processing_queue.append(
                            {'name': new_meeting_name, 'file_obj': uploaded_m4a_file}
                        )
                        st.session_state.show_upload_form = False
                        st.success(f"会议 “{new_meeting_name}” 已加入处理队列。")
                        st.rerun()
    st.markdown("---")

    if st.session_state.processing_queue:
        item_to_process = st.session_state.processing_queue.pop(0)
        meeting_to_process_name = item_to_process['name']
        uploaded_file_for_processing = item_to_process['file_obj']
        if meeting_to_process_name in st.session_state.uploaded_meetings and \
           st.session_state.uploaded_meetings[meeting_to_process_name]['status'] == 'queued':
            with st.spinner(f"正在处理会议 “{meeting_to_process_name}”... 请稍候..."):
                process_meeting_file(meeting_to_process_name, uploaded_file_for_processing)

    if not st.session_state.uploaded_meetings:
        st.caption("暂无会话记录。")
    else:
        st.markdown("#### 已有会议纪要")
        # Sort by name for consistent display
        sorted_meeting_names = sorted(st.session_state.uploaded_meetings.keys())

        for name in sorted_meeting_names:
            data = st.session_state.uploaded_meetings[name]
            status = data.get('status', '未知状态')
            is_processed = (status == 'processed')
            button_text = name
            if not is_processed:
                button_text = f"{name} ({status})"

            button_type = "primary" if st.session_state.show_summary_for_meeting_name == name else "secondary"
            if st.button(button_text, key=f"btn_show_summary_{name}", use_container_width=True,
                         type=button_type,
                         disabled=not is_processed and not status.startswith("错误")):
                if is_processed or status.startswith("错误"):
                    st.session_state.show_summary_for_meeting_name = name
                    st.rerun()

# --- Main Panel ---
main_header_cols = st.columns([0.03, 0.97])
with main_header_cols[0]:
    st.markdown("### ☰") # Placeholder for potential future sidebar toggle icon
with main_header_cols[1]:
    st.markdown("### 多Agent会议纪要与聊天系统 (RAG版)")
st.markdown("---")

# --- Summary "Modal" Display Logic ---
if st.session_state.get('show_summary_for_meeting_name'):
    meeting_name_for_summary = st.session_state.show_summary_for_meeting_name
    meeting_data = st.session_state.uploaded_meetings.get(meeting_name_for_summary, {})
    summary_content_display = "摘要加载中..."
    summary_title_snippet_display = meeting_name_for_summary

    if meeting_data.get('status', '').startswith("错误"):
        summary_content_display = f"处理会议 “{meeting_name_for_summary}” 时发生错误。\n状态: {meeting_data['status']}"
    elif 'summary_content' in meeting_data:
        summary_content_display = meeting_data['summary_content']
    elif 'summary_file_path' in meeting_data and os.path.exists(meeting_data['summary_file_path']):
        try:
            with open(meeting_data['summary_file_path'], "r", encoding="utf-8") as f:
                summary_content_display = f.read()
            st.session_state.uploaded_meetings[meeting_name_for_summary]['summary_content'] = summary_content_display
        except Exception as e:
            summary_content_display = f"错误：加载摘要文件失败。\n{e}"
    else:
        summary_content_display = "错误：找不到摘要文件或内容。"

    if meeting_data.get('status') != 'processed' and not meeting_data.get('status', '').startswith("错误"):
         summary_content_display = f"会议 “{meeting_name_for_summary}” 正在处理或状态未知 ({meeting_data.get('status', '未知')})。摘要尚不可用。"

    first_line_of_summary = summary_content_display.split('\n', 1)[0]
    summary_title_snippet_display = first_line_of_summary[:50] + "..." if len(first_line_of_summary) > 50 else first_line_of_summary

    st.markdown(f"#### 会议摘要: {meeting_name_for_summary}")
    if meeting_data.get('status') == 'processed':
        st.markdown(f"##### \"{summary_title_snippet_display}\"")
    st.text_area("摘要内容", summary_content_display, height=300, key=f"summary_modal_text_{meeting_name_for_summary}", disabled=True)

    btn_cols_dialog = st.columns([1, 1.5, 3]) if meeting_data.get('status') == 'processed' else st.columns([1,3])
    with btn_cols_dialog[0]:
        if st.button("关闭摘要", key=f"close_modal_{meeting_name_for_summary}", use_container_width=True):
            st.session_state.show_summary_for_meeting_name = None
            st.rerun()
    if meeting_data.get('status') == 'processed':
        with btn_cols_dialog[1]:
            if st.button("以此会议开始对话", type="primary", key=f"chat_modal_{meeting_name_for_summary}", use_container_width=True):
                st.session_state.selected_meetings_for_chat = [meeting_name_for_summary]
                st.session_state.show_summary_for_meeting_name = None
                st.rerun()
    st.markdown("---")

# --- Chat Interface or Welcome Message (if no summary modal) ---
elif not st.session_state.show_summary_for_meeting_name:
    st.subheader("会议对话 (RAG)")

    processed_meetings_options = [
        name for name, data in st.session_state.uploaded_meetings.items()
        if data.get('status') == 'processed' and data.get('rag_data_file_path') # Ensure RAG data exists
    ]

    if not processed_meetings_options:
        st.info("没有已处理并生成RAG数据的会议可供选择对话。请先上传并成功处理会议。")
    else:
        selected_chat_meetings = st.multiselect(
            "选择参与对话的会议纪要 (可多选):",
            options=processed_meetings_options,
            default=st.session_state.selected_meetings_for_chat,
            key="chat_multi_select_rag"
        )
        if selected_chat_meetings != st.session_state.selected_meetings_for_chat:
            st.session_state.selected_meetings_for_chat = selected_chat_meetings
            st.rerun()

        if st.session_state.selected_meetings_for_chat:
            chat_key = tuple(sorted(st.session_state.selected_meetings_for_chat))
            if chat_key not in st.session_state.chat_history:
                st.session_state.chat_history[chat_key] = []

            st.markdown(f"#### 对话记录 (RAG): {', '.join(st.session_state.selected_meetings_for_chat)}")
            for message in st.session_state.chat_history[chat_key]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            if prompt_chat := st.chat_input(f"针对 “{', '.join(st.session_state.selected_meetings_for_chat)}” (RAG模式) 提问..."):
                st.session_state.chat_history[chat_key].append({"role": "user", "content": prompt_chat})
                with st.chat_message("user"):
                    st.markdown(prompt_chat)

                with st.chat_message("assistant"):
                    message_placeholder_chat = st.empty()
                    with st.spinner("AI (RAG) 思考中..."):
                        # --- RAG Retrieval and LLM Call ---
                        full_response_chat = get_chat_response_rag(
                            prompt_chat,
                            st.session_state.selected_meetings_for_chat
                        )
                        # --- End RAG ---

                    streamed_response_chat = ""
                    # Simulate streaming for better UX
                    for char_chat in full_response_chat:
                        streamed_response_chat += char_chat
                        message_placeholder_chat.markdown(streamed_response_chat + "▌")
                        time.sleep(0.005) # Adjust for desired speed
                    message_placeholder_chat.markdown(streamed_response_chat)

                st.session_state.chat_history[chat_key].append({"role": "assistant", "content": full_response_chat})
                st.rerun()
        else:
            st.info("请从上方选择一个或多个已处理的会议纪要以开始RAG对话。")
            st.text_input("请输入你的问题", disabled=True, key="main_chat_disabled_rag_placeholder", placeholder="选择会议后可在此输入")

else:
    st.header("欢迎使用会议纪要对话系统 (RAG版)")
    st.info("请从左侧选择一个已处理的会议查看摘要，或通过摘要页进入对话。您也可以直接在对话区选择会议。")
    st.text_input("请输入你的问题", disabled=True, key="main_disabled_input_rag_placeholder_welcome", placeholder="选择一个会话后可在此输入")

st.caption("以上内容为人工智能生成，任何在本文出现的信息（包括但不限于预测、图表、任何形式的表述等）均只作为参考。")