from openai import OpenAI
with open(r'prompt/deepseek_apikey.txt') as f:
    apikey = f.read().strip()
client = OpenAI(api_key=apikey, base_url="https://api.deepseek.com")
messages = [
    # {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "给几个英文词汇，表达'使用'的意思"},
]
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    # stream=False
)
# https://api-docs.deepseek.com/zh-cn/api/create-completion 查看 response 结构
content = response.choices[0].message.content
print(content)
result = response.model_dump_json(indent=2)
with open('test2-1.json', 'w', encoding='utf-8') as f:
    f.write(result)

# 上下文 (多轮对话) 参考 https://api-docs.deepseek.com/zh-cn/guides/reasoning_model
messages.append({'role': 'assistant', 'content': content})
messages.append({'role': 'user', 'content': '再给多几个'})
response2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages
)
print(response2.choices[0].message.content)
result2 = response2.model_dump_json(indent=2)
with open('test2-2.json', 'w', encoding='utf-8') as f:
    f.write(result2)

# 官方 system role 和 json 输出 https://api-docs.deepseek.com/zh-cn/guides/json_mode
# few-shot https://api-docs.deepseek.com/zh-cn/guides/kv_cache