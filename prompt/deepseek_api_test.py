from openai import OpenAI
with open(r'prompt/deepseek_apikey.txt') as f:
    apikey = f.read().strip()
client = OpenAI(api_key=apikey, base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "介绍LLM API调用里system role的作用，与user role的区别，举例如何设置system role，以deepseek为例"},
    ],
    stream=False
)
# https://api-docs.deepseek.com/zh-cn/api/create-completion 查看 response 结构
print(response.choices[0].message.content)
result = response.model_dump_json(indent=2)
with open('test1.json', 'w', encoding='utf-8') as f:
    f.write(result)