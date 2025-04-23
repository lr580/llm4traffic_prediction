from openai import OpenAI
with open(r'prompt/deepseek_apikey.txt') as f:
    apikey = f.read().strip()
client = OpenAI(api_key=apikey, base_url="https://api.deepseek.com")
response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "介绍一下使用api调用deepseek返回的response的结构"},
    ],
    stream=False
)
# https://api-docs.deepseek.com/zh-cn/api/create-completion 查看 response 结构
print(response.choices[0].message.content)
