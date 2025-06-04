import json
import requests
import re
# os and pathlib are not needed
# from datetime import datetime # Not needed

class MeetingAnalyzerV2:
    def __init__(self):
        self.api_key = open('qwen.key', 'r').read()
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        # Increased max_tokens as summaries can be long, especially combined ones
        self.default_parameters = {"temperature": 0.3, "max_tokens": 3500} 

    def detect_language(self, text):
        # Simple language detection based on character counts
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        if chinese_chars > english_chars:
            return "中文"
        elif english_chars > chinese_chars and english_chars > 20:
            return "英文"
        elif chinese_chars > 0 :
            return "中文"
        else:
            return "英文"

    def call_ai(self, prompt, parameters=None):
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        current_parameters = self.default_parameters.copy()
        if parameters:
            current_parameters.update(parameters)
            
        data = {
            "model": "qwen-turbo", # or qwen-long for very long contexts if needed/available
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": current_parameters
        }
        
        try:
            # print(f"🔄 Calling AI with prompt (first 100 chars): {prompt[:100]}...")
            resp = requests.post(self.api_url, headers=headers, json=data, timeout=120) # Increased timeout
            # print(f"📡 API Status: {resp.status_code}")
            
            if resp.status_code == 200:
                result = resp.json()
                if 'output' in result:
                    if 'choices' in result['output'] and result['output']['choices']:
                        return result['output']['choices'][0]['message']['content']
                    elif 'text' in result['output']:
                        return result['output']['text']
                print(f"⚠️ Unexpected API response structure: {result}")
                return None
            else:
                print(f"❌ API Error: {resp.status_code} - {resp.text}")
                return f"API_ERROR: {resp.status_code} - {resp.text}" # Return error string
        except requests.exceptions.RequestException as e:
            print(f"❌ Request Exception: {str(e)}")
            return f"REQUEST_EXCEPTION: {str(e)}"
        except Exception as e:
            print(f"❌ AI Call Exception: {str(e)}")
            return f"AI_CALL_EXCEPTION: {str(e)}"
        return None


    def _split_text_into_chunks(self, text: str, max_words: int = 3000) -> list[str]:
        """Splits text into chunks of approximately max_words."""
        words = text.split() # Simple split by whitespace
        chunks = []
        for i in range(0, len(words), max_words):
            chunk_words = words[i:i + max_words]
            chunks.append(" ".join(chunk_words))
        # print(f"Split text into {len(chunks)} chunks.")
        return chunks

    def _summarize_chunk_in_language(self, chunk_text: str, is_chinese_prompt: bool) -> str:
        """Summarizes a single chunk of text using Markdown format."""
        if is_chinese_prompt:
            prompt = f"""
你是一个会议纪要助手，请你根据以下模版生成一个会议纪要
            
会议纪要 (Meeting Minutes)

会议名称：[根据输入填写]

一、会议议题与讨论摘要：

    1.  议题一：[议题名称]
        *   主要讨论点：
            *   [讨论点A的简述]
            *   [讨论点B的简述]
            *   [不同观点或重要发言摘要]
        *   决议：
            *   [决议1的内容]
        *   行动项：
            *   [负责人A]：[具体任务A] - [截止日期A]

    2.  议题二：[议题名称]
        *   主要讨论点：
            *   ...
        *   决议：
            *   ...
        *   行动项：
            *   ...

    (根据实际议题数量增减)

二、总结与关键决策：
    *   [对会议达成的主要共识或重要决策进行总结]

三、行动计划与分配 (Action Items Summary):
    *   行动项1：
        *   任务描述：[具体任务]
        *   负责人：[姓名]
        *   截止日期：[YYYY-MM-DD]
    *   行动项2：
        *   任务描述：[具体任务]
        *   负责人：[姓名]
        *   截止日期：[YYYY-MM-DD]
    *   ...

四、其他事项 (Any Other Business - AOB)：
    *   [记录讨论到的其他未列入议程但重要事项]
    
- Original text content: {chunk_text}
"""
        else: # English prompt
            prompt = f"""
Meeting Minutes

Meeting Title: [Based on input]


I. Agenda Items & Discussion Summary:

    1.  Agenda Item 1: [Item Title]
        *   Key Discussion Points:
            *   [Brief description of discussion point A]
            *   [Brief description of discussion point B]
            *   [Summary of different opinions or important statements]
        *   Decisions:
            *   [Content of decision 1]
        *   Action Items:
            *   [Owner A]: [Specific Task A] - [Due Date A]

    2.  Agenda Item 2: [Item Title]
        *   Key Discussion Points:
            *   ...
        *   Decisions:
            *   ...
        *   Action Items:
            *   ...

    (Add or remove sections based on the number of agenda items)

II. Summary & Key Decisions:
    *   [Summarize the main consensus or important decisions reached during the meeting]

III. Action Items Summary & Assignments:
    *   Action Item 1:
        *   Task Description: [Specific Task]
        *   Responsible: [Name]
        *   Due Date: [YYYY-MM-DD]
    *   Action Item 2:
        *   Task Description: [Specific Task]
        *   Responsible: [Name]
        *   Due Date: [YYYY-MM-DD]
    *   ...

IV. Any Other Business (AOB):
    *   [Record any other important matters discussed that were not on the agenda]
    
- Original text content: {chunk_text}
"""
        # print(f"Summarizing chunk (lang: {'CH' if is_chinese_prompt else 'EN'})...")
        summary = self.call_ai(prompt)
        if summary and not (summary.startswith("API_ERROR:") or summary.startswith("REQUEST_EXCEPTION:") or summary.startswith("AI_CALL_EXCEPTION:")):
            return summary.strip()
        else:
            # print(f"Failed to summarize chunk or AI returned error: {summary}")
            error_msg_key = "主题：分块摘要失败" if is_chinese_prompt else "**Topic:** Chunk Summary Failed"
            error_msg_val = f"未能处理此文本块。错误：{summary if summary else '未知错误'}" if is_chinese_prompt else f"Could not process this text block. Error: {summary if summary else 'Unknown error'}"
            return f"{error_msg_key}\n**具体内容：** {error_msg_val}"


    def _consolidate_summaries_in_language(self, combined_summaries: str, is_chinese_prompt: bool) -> str:
        """Consolidates and deduplicates combined chunk summaries."""
        if is_chinese_prompt:
            prompt = f"""
你是一位专业的编辑，擅长整合和提炼信息。
以下是多个文本片段的初步概括，每个概括都包含“主题”和“具体内容”。
你的任务是：
1. 仔细阅读所有概括。
2. 识别并合并重复或高度相似的主题。
3. 对每个独特或合并后的主题，生成一个最终的、精炼的总结。


待整合的概括内容如下：
---
{combined_summaries}
---
请开始整合和提炼。
"""
        else: # English prompt
            prompt = f"""
You are an expert editor skilled in synthesizing and refining information.
Below are preliminary summaries from multiple text segments, each containing a "Topic" and "Content".
Your tasks are to:
1. Carefully read all the summaries.
2. Identify and merge duplicate or highly similar topics.
3. For each unique or merged topic, generate a final, concise summary.

Summaries to consolidate:
---
{combined_summaries}
---
Please begin consolidation and refinement.
"""
        # print(f"Consolidating summaries (lang: {'CH' if is_chinese_prompt else 'EN'})...")
        # Use higher max_tokens for consolidation
        final_result = self.call_ai(prompt, parameters={"max_tokens": 8000}) # qwen-turbo max is 8k tokens for context
        if final_result and not (final_result.startswith("API_ERROR:") or final_result.startswith("REQUEST_EXCEPTION:") or final_result.startswith("AI_CALL_EXCEPTION:")):
            return final_result.strip()
        else:
            # print(f"Failed to consolidate summaries or AI returned error: {final_result}")
            return f"### 整合失败\n无法完成最终摘要整合。原始合并摘要：\n{combined_summaries}\n错误：{final_result if final_result else '未知错误'}" \
                if is_chinese_prompt \
                else f"### Consolidation Failed\nCould not complete final summary consolidation. Original combined summaries:\n{combined_summaries}\nError: {final_result if final_result else 'Unknown error'}"


    def analyze(self, meeting_text: str) -> str:
        """
        Analyzes meeting text by splitting, summarizing chunks, and consolidating.
        Returns a Markdown string.
        """
        if not meeting_text or not meeting_text.strip():
            return "### 错误\n输入的会议内容为空。"

        language = self.detect_language(meeting_text)
        is_chinese = (language == "中文")
        # print(f"🌐 Detected language: {language}")

        # 1. Split text into chunks
        # Word count can be tricky for CJK languages. 3000 words might be ~6000-9000 characters.
        # qwen-turbo has a context window of 8k tokens. Let's use a character limit for splitting
        # to be safer, or stick to word count and hope the model handles tokenization well.
        # For words, 3000 words is a good number.
        chunks = self._split_text_into_chunks(meeting_text, max_words=2800) # Slightly less than 3000 for safety margin

        if not chunks:
            return "### 错误\n无法将文本分割成块。" if is_chinese else "### Error\nCould not split text into chunks."

        # 2. Summarize each chunk
        chunk_summaries = []
        for i, chunk in enumerate(chunks):
            # print(f"Processing chunk {i+1}/{len(chunks)}")
            summary = self._summarize_chunk_in_language(chunk, is_chinese)
            chunk_summaries.append(summary)
        
        combined_chunk_summaries = "\n\n---\n\n".join(chunk_summaries) # Separator for AI to distinguish

        # 3. Consolidate and deduplicate summaries
        final_markdown_summary = self._consolidate_summaries_in_language(combined_chunk_summaries, is_chinese)
        
        return final_markdown_summary


def analyze_meeting_text_v2(meeting_content: str) -> str:
    """
    Main function to analyze meeting text (V2: Markdown multi-stage).
    
    Args:
        meeting_content: The raw text of the meeting.
        api_key: Your Dashscope API key.
        
    Returns:
        A Markdown string of the analysis result, or an error message.
    """
    if not meeting_content or not meeting_content.strip():
        return "### 错误\n会议内容不能为空。"

    analyzer = MeetingAnalyzerV2()
    analysis_result_markdown = analyzer.analyze(meeting_content)
    
    return analysis_result_markdown

if __name__ == "__main__":
    analyzer = MeetingAnalyzerV2()
    analysis_result_markdown = analyzer.analyze(open('/home/dell/mwy/AIML-project/sora.txt', 'r').read())
    print(analysis_result_markdown)
    