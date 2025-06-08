import json
import re
import requests
from typing import List, Dict, Any, Optional

class MeetingAnalyzer:
    """
    一个用于分析会议记录文本的类。

    该类可以自动检测文本语言，将长文本分块，调用AI模型（通义千问）
    对每个分块进行摘要，最后将所有分块的摘要整合成一份遵循
    《罗伯特议事规则》的正式会议纪要。

    属性:
        API_URL (str): 阿里Dashscope服务的API端点。
        API_MODEL (str): 使用的AI模型名称。
        DEFAULT_PARAMS (Dict[str, Any]): 调用AI时的默认参数。
        api_key (str): 从文件中读取的API密钥。
    """

    API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    API_MODEL = "qwen-turbo"
    DEFAULT_PARAMS = {"temperature": 0.3, "max_tokens": 3500}
    CHUNK_MAX_WORDS = 2800

    def __init__(self, api_key_path: str = 'qwen.key'):
        """
        初始化 MeetingAnalyzer 实例。

        Args:
            api_key_path (str): 存储API密钥的文件路径。

        Raises:
            FileNotFoundError: 如果API密钥文件不存在。
            ValueError: 如果API密钥文件为空。
        """
        try:
            with open(api_key_path, 'r') as f:
                self.api_key = f.read().strip()
            if not self.api_key:
                raise ValueError(f"API key file '{api_key_path}' is empty.")
        except FileNotFoundError:
            print(f"Error: API key file not found at '{api_key_path}'.")
            raise
        except Exception as e:
            print(f"An error occurred while reading the API key: {e}")
            raise

    def _call_ai(self, prompt: str, params: Optional[Dict[str, Any]] = None) -> str:
        """
        向AI模型发送请求并获取响应。

        Args:
            prompt (str): 发送给AI的完整提示。
            params (Optional[Dict[str, Any]]): 此次调用特定的API参数，可覆盖默认值。

        Returns:
            str: AI模型的响应内容。如果发生错误，则返回一个包含错误信息的字符串。
        """
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        current_params = self.DEFAULT_PARAMS.copy()
        if params:
            current_params.update(params)
            
        data = {
            "model": self.API_MODEL,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": current_params
        }
        
        try:
            response = requests.post(self.API_URL, headers=headers, json=data, timeout=120)
            response.raise_for_status()

            result = response.json()
            output = result.get('output', {})
            
            if 'choices' in output and output['choices']:
                return output['choices'][0]['message']['content']
            elif 'text' in output:
                return output['text']
            
            print(f"⚠️ Unexpected API response structure: {result}")
            return "ERROR: Unexpected API response structure"

        except requests.exceptions.RequestException as e:
            print(f"❌ Request Exception: {e}")
            return f"ERROR: Request Exception - {e}"
        except Exception as e:
            print(f"❌ AI Call Exception: {e}")
            return f"ERROR: AI Call Exception - {e}"

    @staticmethod
    def _detect_language(text: str) -> str:
        """
        根据文本中中英文字符的数量，简单判断文本的主要语言。

        Args:
            text (str): 待检测的文本。

        Returns:
            str: "中文" 或 "英文"。
        """
        chinese_chars = re.findall(r'[\u4e00-\u9fff]', text)
        english_chars = re.findall(r'[a-zA-Z]', text)
        
        if len(chinese_chars) > len(english_chars):
            return "中文"
        elif len(english_chars) > len(chinese_chars) and len(english_chars) > 20:
            return "英文"
        elif len(chinese_chars) > 0:
            return "中文"
        else:
            return "英文"

    @staticmethod
    def _split_text_into_chunks(text: str, max_words: int) -> List[str]:
        """
        将文本按大致的单词数量分割成多个块。

        Args:
            text (str): 原始文本。
            max_words (int): 每个块的最大单词数。

        Returns:
            List[str]: 分割后的文本块列表。
        """
        words = text.split()
        if not words:
            return []
        
        chunks = [" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words)]
        print(f"Text split into {len(chunks)} chunk(s).")
        return chunks

    def _get_summarize_prompt(self, chunk_text: str, is_chinese: bool) -> str:
        """
        根据语言生成用于摘要单个文本块的提示。

        Args:
            chunk_text (str): 会议文本的一个分块。
            is_chinese (bool): 如果目标语言是中文，则为True。

        Returns:
            str: 格式化后的完整提示。
        """
        if is_chinese:
            return f"""
- 你是一个一丝不苟且立场中立的会议记录员。你的任务是严格遵循《罗伯特议事规则》的原则，生成正式的会议记录。会议记录必须清晰、简洁、准确地记录会议上“做了什么”，而不是“说了什么”。

- 操作指南：
1.  **基本信息**：在纪要开头包含组织名称、会议类型、日期、开始时间和地点。
2.  **出席情况**：列出所有出席、缺席的成员，并注明会议达到法定人数、主持人及记录员身份。
3.  **批准上次会议记录**：说明上次会议记录是否被宣读和通过，并记录任何修正。
4.  **报告**：记录所有提交的报告，包括报告人姓名及对报告采取的行动。
5.  **动议 (Motions)**：对于每一项动议，精确记录其确切措辞、动议人、表决结果（通过/未通过）以及具体的票数统计（赞成、反对、弃权）。
6.  **休会 (Adjournment)**：记录会议的休会时间。
7.  **语言与格式**：使用正式、客观的过去时态。按照会议议程的逻辑顺序组织会议记录。

- 会议文字记录如下: 
---
{chunk_text}
---
请根据以上规则生成正式的会议记录。
"""
        else:
            return f"""
- You are a meticulous and impartial Meeting Minutes Agent. Your task is to generate formal meeting minutes that strictly adhere to the principles of Robert's Rules of Order. The minutes must be a clear, concise, and accurate record of what was *done* at the meeting, not what was *said*.

- Instructions:
1.  **Essential Information**: Begin with the organization name, meeting type, date, start time, and location.
2.  **Attendance**: List all members present and absent. Note the presence of a quorum and identify the presiding officer and secretary.
3.  **Approval of Previous Minutes**: State if the previous minutes were read and approved, and record any corrections.
4.  **Reports**: Document any reports presented, including the presenter's name and any action taken.
5.  **Motions**: For every motion, record its exact wording, the mover's name, the outcome of the vote (e.g., carried, failed), and the vote count (in favor, against, abstentions).
6.  **Adjournment**: Record the time the meeting was adjourned.
7.  **Language and Format**: Use formal, objective, past-tense language. Organize the minutes logically according to the meeting agenda.

- Original text content to be processed:
---
{chunk_text}
---
Please generate the formal meeting minutes based on the rules above.
"""

    def _summarize_chunk(self, chunk_text: str, is_chinese: bool) -> str:
        """
        对单个文本块进行摘要。

        Args:
            chunk_text (str): 待摘要的文本块。
            is_chinese (bool): 是否使用中文提示。

        Returns:
            str: 摘要结果或错误信息。
        """
        prompt = self._get_summarize_prompt(chunk_text, is_chinese)
        summary = self._call_ai(prompt)
        
        if summary.startswith("ERROR:"):
            error_topic = "分块摘要失败" if is_chinese else "Chunk Summary Failed"
            error_content = f"未能处理此文本块。错误：{summary}" if is_chinese else f"Could not process this text block. Error: {summary}"
            return f"**{error_topic}**\n\n{error_content}"
            
        return summary.strip()

    def _get_consolidation_prompt(self, combined_summaries: str, is_chinese: bool) -> str:
        """
        生成用于整合多个摘要的提示。

        Args:
            combined_summaries (str): 由分隔符连接的多个分块摘要。
            is_chinese (bool): 是否使用中文提示。

        Returns:
            str: 格式化后的整合提示。
        """
        if is_chinese:
            return f"""
你是一位专业的编辑，擅长整合与提炼信息。
以下是同一场会议不同部分的会议纪要初稿，由分隔符 `---` 隔开。
你的任务是：
1.  仔细阅读所有片段。
2.  将它们整合成一份 **单一、完整且连贯** 的会议纪要。
3.  **消除重复**：合并重复的条目（如基本信息、出席名单等），确保每个信息只出现一次。
4.  **按逻辑排序**：确保最终的纪要遵循标准的会议流程（基本信息 -> 出席情况 -> 批准上次纪要 -> 报告 -> 动议 -> 休会）。
5.  保持《罗伯特议事规则》所要求的正式、客观的风格。

待整合的纪要片段如下：
---
{combined_summaries}
---
请输出一份整合、去重并按逻辑排序后的最终会议纪要。
"""
        else:
            return f"""
You are an expert editor skilled in synthesizing and refining information.
Below are several draft sections of minutes from the same meeting, separated by `---`.
Your tasks are to:
1.  Carefully read all segments.
2.  Consolidate them into a **single, cohesive, and complete** set of meeting minutes.
3.  **Eliminate Redundancy**: Merge duplicate entries (like header information, attendance lists, etc.) to ensure each piece of information appears only once.
4.  **Ensure Logical Order**: Organize the final minutes to follow a standard meeting flow (Essential Info -> Attendance -> Approval of Minutes -> Reports -> Motions -> Adjournment).
5.  Maintain the formal, objective style required by Robert's Rules of Order.

The draft minute segments to be consolidated are below:
---
{combined_summaries}
---
Please produce the final, consolidated, deduplicated, and logically ordered meeting minutes.
"""

    def _consolidate_summaries(self, chunk_summaries: List[str], is_chinese: bool) -> str:
        """
        将多个分块摘要整合成一份最终的会议纪要。

        Args:
            chunk_summaries (List[str]): 包含各分块摘要的列表。
            is_chinese (bool): 是否使用中文提示。

        Returns:
            str: 最终的、整合后的会议纪要或错误信息。
        """
        combined_text = "\n\n---\n\n".join(chunk_summaries)
        prompt = self._get_consolidation_prompt(combined_text, is_chinese)
        
        consolidation_params = {"max_tokens": 8000}
        final_result = self.call_ai(prompt, params=consolidation_params)
        
        if final_result.startswith("ERROR:"):
            error_header = "### 整合失败" if is_chinese else "### Consolidation Failed"
            error_message = (f"无法完成最终摘要整合。错误：{final_result}"
                             if is_chinese
                             else f"Could not complete final summary consolidation. Error: {final_result}")
            return f"{error_header}\n{error_message}\n\n**原始合并摘要:**\n{combined_text}"
        
        return final_result.strip()

    def analyze(self, meeting_text: str) -> str:
        """
        执行完整的会议文本分析流程。

        流程包括：语言检测 -> 文本分块 -> 分块摘要 -> 整合摘要。

        Args:
            meeting_text (str): 完整的会议记录原始内容。

        Returns:
            str: Markdown 格式的最终会议纪要。
        """
        if not meeting_text or not meeting_text.strip():
            return "### 错误\n输入的会议内容为空。"

        language = self._detect_language(meeting_text)
        is_chinese = (language == "中文")
        print(f"🌐 Detected language: {language}")

        chunks = self._split_text_into_chunks(meeting_text, self.CHUNK_MAX_WORDS)
        if not chunks:
            return "### 错误\n无法将文本分割成块。" if is_chinese else "### Error\nCould not split text into chunks."

        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            print(f"Processing chunk {i + 1}/{len(chunks)}...")
            summary = self._summarize_chunk(chunk, is_chinese)
            chunk_summaries.append(summary)
        
        if len(chunk_summaries) == 1:
            return chunk_summaries[0]
            
        print("Consolidating all summaries...")
        final_summary = self._consolidate_summaries(chunk_summaries, is_chinese)
        
        return final_summary


def analyze_meeting_from_file(file_path: str, api_key_path: str = 'qwen.key') -> str:
    """
    从文件中读取会议内容并进行分析的辅助函数。

    Args:
        file_path (str): 包含会议内容的文本文件路径。
        api_key_path (str): API密钥文件路径。

    Returns:
        str: 分析结果或错误信息。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        return f"### 错误\n找不到文件：{file_path}"
    except Exception as e:
        return f"### 错误\n读取文件时出错：{e}"

    if not content or not content.strip():
        return "### 错误\n文件内容为空。"
        
    analyzer = MeetingAnalyzer(api_key_path=api_key_path)
    return analyzer.analyze(content)


# --- 主执行入口 ---
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze a meeting transcript and generate formal minutes.")
    parser.add_argument("input_file", type=str, help="The path to the text file containing the meeting transcript.")
    parser.add_argument("--key_file", type=str, default="qwen.key", help="The path to the file containing the API key.")
    parser.add_argument("--output_file", type=str, help="The path to save the generated minutes (optional).")
    
    args = parser.parse_args()
    
    print(f"Starting analysis for '{args.input_file}'...")
    analysis_result = analyze_meeting_from_file(args.input_file, args.key_file)
    
    print("\n--- ANALYSIS RESULT ---\n")
    print(analysis_result)
    
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                f.write(analysis_result)
            print(f"\n✅ Result successfully saved to '{args.output_file}'.")
        except Exception as e:
            print(f"\n❌ Error saving result to file: {e}")