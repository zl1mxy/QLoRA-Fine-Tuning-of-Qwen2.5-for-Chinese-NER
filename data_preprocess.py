import pandas as pd
import json
import re
from datasets import Dataset, DatasetDict
import numpy as np

def load_ner_data():
    """加载NER数据集"""
    print("加载NER数据集...")
    
    # 训练集
    train_df = pd.read_parquet("ner_output/ner_data.parquet")
    print(f"训练集: {len(train_df):,} 行")
    
    # 验证集
    valid_df = pd.read_parquet("ner_datasets/valid_ner/valid_ner.parquet")
    print(f"验证集: {len(valid_df):,} 行")
    
    return train_df, valid_df

def format_instruction_data(row):
    """格式化指令数据"""
    # 提取instruction中的核心指令
    instruction = row['instruction']
    if isinstance(instruction, str) and len(instruction) > 500:
        # 截断过长的指令，保留核心部分
        lines = instruction.split('\n')
        core_instruction = lines[0] if lines else instruction[:200]
        instruction = core_instruction
    
    # 构建完整的prompt
    prompt = f"""请识别并标注以下中文文本中的命名实体（包括人名、地名、时间、组织名、公司名、产品名等）。

文本：
{row['input']}

请按照以下格式标注：
{{实体类型: 实体文本}}

请直接输出标注结果："""
    
    # 获取output，确保格式正确
    if isinstance(row['output'], str) and '{{' in row['output'] and '}}' in row['output']:
        response = row['output']
    else:
        # 如果没有正确的格式，使用input作为占位
        response = f"未找到实体标注。原始文本：{row['input'][:100]}..."
    
    return {
        "instruction": instruction[:300] if isinstance(instruction, str) else "",
        "input": row['input'],
        "output": response,
        "text": f"### 指令：{instruction}\n\n### 输入：{row['input']}\n\n### 输出：{response}"
    }

def create_huggingface_dataset():
    """创建Hugging Face数据集"""
    train_df, valid_df = load_ner_data()
    
    print("\n格式化训练数据...")
    train_data = []
    for _, row in train_df.iterrows():
        try:
            formatted = format_instruction_data(row)
            train_data.append(formatted)
        except Exception as e:
            print(f"格式化训练数据时出错: {e}")
            continue
    
    print("格式化验证数据...")
    valid_data = []
    for _, row in valid_df.iterrows():
        try:
            formatted = format_instruction_data(row)
            valid_data.append(formatted)
        except Exception as e:
            print(f"格式化验证数据时出错: {e}")
            continue
    
    # 转换为Dataset
    train_dataset = Dataset.from_list(train_data)
    valid_dataset = Dataset.from_list(valid_data)
    
    # 创建DatasetDict
    dataset_dict = DatasetDict({
        "train": train_dataset,
        "validation": valid_dataset
    })
    
    print(f"\n✅ 数据集创建完成:")
    print(f"   训练集: {len(train_dataset):,} 样本")
    print(f"   验证集: {len(valid_dataset):,} 样本")
    
    # 保存到磁盘
    dataset_dict.save_to_disk("ner_instruction_dataset")
    print("💾 数据集已保存到: ner_instruction_dataset")
    
    # 保存为JSONL供检查
    with open("ner_dataset_sample.jsonl", "w", encoding="utf-8") as f:
        for i in range(min(10, len(train_data))):
            f.write(json.dumps(train_data[i], ensure_ascii=False) + "\n")
    print("📝 样本数据已保存到: ner_dataset_sample.jsonl")
    
    return dataset_dict

def analyze_entity_distribution(dataset):
    """分析实体分布"""
    print("\n🔍 分析实体类型分布...")
    
    entity_types = []
    for sample in dataset["train"].select(range(min(1000, len(dataset["train"])))):
        output = sample["output"]
        if isinstance(output, str):
            matches = re.findall(r'\{\{([^:]+):', output)
            entity_types.extend(matches)
    
    if entity_types:
        from collections import Counter
        type_counter = Counter(entity_types)
        
        print("实体类型分布 (前20):")
        for entity_type, count in type_counter.most_common(20):
            print(f"  {entity_type:20s}: {count:4} 次")
    
    return entity_types

if __name__ == "__main__":
    dataset = create_huggingface_dataset()
    analyze_entity_distribution(dataset)