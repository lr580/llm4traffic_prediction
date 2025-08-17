from openai import OpenAI
import os
import json
import requests
class CacheQuery():
    '''如果没有询问过，就调用API询问，如果询问过，返回询问结果。\n
    无上下文，不使用流。'''
    def __init__(self, path=''):
        # self.client = OpenAI(api_key="key", base_url="https://api.xxx.com")
        self.client = None # 子类实现
        self.model = 'LLM' # 子类实现
        self.path = path
        if self.path:
            os.makedirs(self.path, exist_ok=True)
            
    def update_path(self, path):
        self.path = path
        os.makedirs(self.path, exist_ok=True)
        
    def query(self, queryID:str, userMessage='', message=[],  systemMessage=''):
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
                
    def makeQuery(self, filepath:str, message:list)->str:
        return '' # to be implememted by subclass

    def extractCache(self, data:dict)->str:
        return '' # to be implememted by subclass
    
    def readCache(self, filepath:str)->str:
        with open(filepath, encoding='utf-8') as f:
            data = json.load(f)
        return self.extractCache(data)
    
with open(r'prompt/deepseek_apikey.txt') as f:
    deepseek_apikey = f.read().strip()
    
class DeepseekQuery(CacheQuery):
    def __init__(self, path='', apikey='', model='deepseek-chat'):
        super().__init__(path)
        if not apikey:
            self.apikey = deepseek_apikey
        self.client = OpenAI(api_key=self.apikey, base_url="https://api.deepseek.com")
        self.model = model 
        """ 'deepseek-chat' or 'deepseek-reasoner' """

    def makeQuery(self, filepath:str, message:list)->str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=message,
            stream=False,
        )
        with open(filepath, 'w', encoding='utf-8') as f:
            result = response.model_dump_json(indent=2)
            f.write(result)
        content = response.choices[0].message.content
        return str(content)

    def extractCache(self, data:dict)->str:
        return data['choices'][0]['message']['content']

class OllamaQuery(CacheQuery):
    def __init__(self, path='', url='http://127.0.0.1:11434/api/chat', model='deepseek-r1:7b'):
        super().__init__(path)
        # self.client = OpenAI(base_url=url, api_key='None')
        self.model = model
        self.url = url

    def makeQuery(self, filepath:str, message:list)->str:
        payload = {
            "model" : self.model,
            "messages": message,
            "stream": False
        }
        response = requests.post(self.url, json=payload)
        response.raise_for_status()
        result = response.json()
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        return self.extractCache(result)

    def extractCache(self, data:dict)->str:
        return data['message']['content']
        
if __name__ == "__main__":
    # q = DeepseekQuery('results/testClass')
    # print(q.query('test001-C', '前端和后端有何区别'))
    q = OllamaQuery('results/testClass')
    # print(q.query('test001-E', '三维前缀和差分的公式是？'))
    print(q.query('test001-G', '我刚刚询问了什么问题')) # 默认无上下文