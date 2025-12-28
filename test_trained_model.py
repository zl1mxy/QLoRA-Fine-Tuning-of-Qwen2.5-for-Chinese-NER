import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. 加载基础模型
print("加载基础模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    "/root/autodl-tmp/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="eager"
)

# 2. 加载LoRA适配器
print("加载LoRA适配器...")
model = PeftModel.from_pretrained(
    base_model,
    "./qwen2.5-7b-ner-qlora-a100-final",
    adapter_name="ner_lora"
)

# 3. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/huggingface_cache/models--Qwen--Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28")

# 4. 测试样本
test_cases = [
    "2023年，马云在杭州阿里巴巴总部发表了讲话。",
    "苹果公司于1976年由史蒂夫·乔布斯创立。",
    "李华博士在北京清华大学教授计算机科学课程。"
]

for i, text in enumerate(test_cases):
    print(f"\n{'='*50}")
    print(f"测试 {i+1}: {text}")
    
    # 构建输入
    prompt = f"""<|im_start|>system
你是一个专业的中文命名实体识别助手。<|im_end|>
<|im_start|>user
请识别以下文本中的实体：

{text}

格式：{{实体类型: 实体}}<|im_end|>
<|im_start|>assistant
"""
    
    # 生成
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=False
        )
    
    # 解码输出
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split("assistant")[-1].strip()
    print(f"识别结果: {answer}")
