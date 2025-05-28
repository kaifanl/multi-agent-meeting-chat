from openai import OpenAI

with open('qwen.key', 'r') as f:
    api_key = f.read().strip()


class APILLMResponse:
    def __init__(self, url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1") -> None:
        self.client = OpenAI(
            api_key=api_key,
            base_url=url
        )
    
    def get_response(self, messages, response_format=None, model_name='qwen-plus'):
        if response_format:
            messages.insert(0, {"role": "system", "content": "Return in JSON format."})
        completion = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2
        )
        answer = completion.choices[0].message.content
        if response_format == "json_object":
            return answer.split("```json")[1].split("```")[0]
        else:
            return answer

if __name__ == "__main__":
    llm_client = APILLMResponse()
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': 'What is the capital of France?'}
    ]
    print(llm_client.get_response(messages))
