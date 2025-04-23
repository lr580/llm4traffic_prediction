当您通过API调用DeepSeek模型时，返回的响应通常遵循标准的结构，便于解析和处理。以下是DeepSeek API返回的典型响应结构及其字段说明：

---

### **1. 基本响应结构**

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "DeepSeek是一个专注于AI技术研发的公司..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 120,
    "total_tokens": 135
  }
}
```

---

### **2. 字段详解**

#### **顶层字段**

- **`id`** (string): 本次请求的唯一标识符。
- **`object`** (string): 对象类型（如 `chat.completion`）。
- **`created`** (integer): 响应生成的时间戳（Unix秒）。
- **`model`** (string): 使用的模型名称（如 `deepseek-chat`）。

#### **`choices` 字段**

返回一个数组（通常仅包含一个元素，除非设置 `n > 1`）：

- **`index`** (integer): 选择的索引（从0开始）。
- **`message`** (object): 模型生成的回复内容：
  - **`role`**: 角色（固定为 `assistant`）。
  - **`content`**: 回复的文本内容。
- **`finish_reason`** (string): 停止生成的原因：
  - `stop`: 正常结束（遇到停止标记）。
  - `length`: 因 `max_tokens` 限制截断。
  - 其他可能原因（如API特定规则）。

#### **`usage` 字段**

- **`prompt_tokens`**: 输入提示（Prompt）消耗的token数。
- **`completion_tokens`**: 生成回复消耗的token数。
- **`total_tokens`**: 总token数（输入+输出）。

---

### **3. 流式响应（Streaming）**

如果调用时启用流式传输（`stream: true`），响应会分多次返回，每个数据块格式如下：

```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion.chunk",
  "created": 1677652288,
  "model": "deepseek-chat",
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "流式输出的片段..."
      },
      "finish_reason": null
    }
  ]
}
```

- **`delta`**: 包含当前流片段的增量内容（非完整消息）。
- **`finish_reason`**: 仅在最后一个数据块中显示停止原因。

---

### **4. 错误响应**

如果调用失败，返回的HTTP状态码非200，响应体包含错误信息：

```json
{
  "error": {
    "code": "invalid_api_key",
    "message": "无效的API密钥",
    "type": "invalid_request_error"
  }
}
```

---

### **5. 注意事项**

1. **字段兼容性**: 结构可能与OpenAI API类似，但具体字段以DeepSeek官方文档为准。
2. **内容安全**: 实际回复的 `content` 字段可能包含Markdown格式或换行符（`\n`）。
3. **调试建议**: 使用工具（如Postman或curl）测试API，观察完整响应。

如果需要更详细的字段说明或示例代码（如Python解析响应），可以参考DeepSeek的官方API文档或告诉我您的具体需求！