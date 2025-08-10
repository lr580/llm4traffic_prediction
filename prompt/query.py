from openai import OpenAI
import os
import json
class CacheQuery():
    '''如果没有询问过，就调用API询问，如果询问过，返回询问结果。\n
    无上下文，不使用流。'''
    def __init__(self, path):
        # self.client = OpenAI(api_key="key", base_url="https://api.xxx.com")
        self.client = None # 子类实现
        self.model = 'LLM' # 子类实现
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        
    def query(self, queryID, userMessage='', message='',  systemMessage=''):
        filepath = os.path.join(self.path, f'{queryID}.json')
        if os.path.exists(filepath):
            content = self.readCache(filepath)
            return content
        if not message:
            message = []
            if systemMessage:
                message.append({"role": "system", "content": systemMessage})
            if userMessage:
                message.append({"role": "user", "content": userMessage})
        content = self.makeQuery(filepath, message)
        return content
                
    def makeQuery(self, filepath, message):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            result = response.model_dump_json(indent=2)
            f.write(result)
        content = response.choices[0].message.content
        return content
    
    def readCache(self, filepath):
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        content = data['choices'][0]['message']['content']
        return content
    
with open(r'prompt/deepseek_apikey.txt') as f:
    deepseek_apikey = f.read().strip()
    
class DeepseekQuery(CacheQuery):
    def __init__(self, path, apikey=''):
        super().__init__(path)
        if not apikey:
            self.apikey = deepseek_apikey
        self.client = OpenAI(api_key=self.apikey, base_url="https://api.deepseek.com")
        self.model = 'deepseek-chat'
        
if __name__ == "__main__":
    q = DeepseekQuery('results/testClass')
    print(q.query('test001-B', '中间件开发和后端有何区别'))