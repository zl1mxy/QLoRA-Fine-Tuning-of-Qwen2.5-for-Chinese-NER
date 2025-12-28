import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import load_from_disk
import os
import wandb

def setup_a100_environment():
    """A100专用环境设置"""
    print("=" * 70)
    print("🚀 A100 40GB + Qwen2.5-7B QLoRA 微调训练")
    print("=" * 70)
    
    # 检查A100
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"✅ GPU: {gpu_name}")
        print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # A100特有优化
        if "A100" in gpu_name:
            print("🎯 A100检测到，启用优化配置:")
            print("   - 使用TF32精度")
            print("   - 增大batch size")
            print("   - 开启梯度检查点")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    
    return "cuda"

def load_model_for_a100(model_name="Qwen/Qwen2.5-7B-Instruct"):
    """A100优化版模型加载"""
    print(f"\n🔧 加载模型: {model_name}")
    
    # A100可使用更高效的量化
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # A100支持bfloat16
        bnb_4bit_use_double_quant=True,
    )
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="right",
        use_fast=False
    )
    
    # 设置pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # 加载模型（A100可用bfloat16加速）
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="eager",
    quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,  # A100优化
    )
    
    model = prepare_model_for_kbit_training(model)
    
    print(f"✅ 模型加载完成")
    print(f"   模型大小: {model.num_parameters() / 1e9:.2f}B 参数")
    print(f"   量化: 4-bit NF4 + bfloat16")
    print(f"   Flash Attention: 已启用")
    
    return model, tokenizer

def setup_a100_lora(model):
    """A100优化的LoRA配置"""
    print("\n🎯 配置LoRA参数（A100优化）")
    
    lora_config = LoraConfig(
        r=32,  # A100显存大，可增加秩
        lora_alpha=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj",
                       "lm_head"],  # 增加lm_head
        lora_dropout=0.05,  # 减少dropout
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # 应用LoRA
    model = get_peft_model(model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🎯 可训练参数: {trainable_params:,} ({(trainable_params/total_params)*100:.2f}%)")
    print(f"🎯 总参数: {total_params:,}")
    
    return model

def train_on_a100():
    """在A100上训练"""
    device = setup_a100_environment()
    
    # 1. 加载数据
    print("\n📂 加载数据集...")
    try:
        dataset = load_from_disk("ner_instruction_dataset")
        print(f"✅ 数据集加载完成:")
        print(f"   训练集: {len(dataset['train']):,} 样本")
        print(f"   验证集: {len(dataset['validation']):,} 样本")
        
        # 显示数据大小
        train_size = len(dataset['train'])
        if train_size < 10000:
            print(f"⚠️  数据量较小 ({train_size} samples)，可增加训练轮数")
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("请先运行: python data_preprocess.py")
        return None
    
    # 2. 加载模型和tokenizer
    model, tokenizer = load_model_for_a100()
    
    # 3. 设置LoRA
    model = setup_a100_lora(model)
    
    # 4. A100优化训练参数
    training_args = TrainingArguments(
        output_dir="./qwen2.5-7b-ner-qlora-a100",
        num_train_epochs=5,  # A100可增加轮数
        per_device_train_batch_size=2,  # A100可增大batch size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,  # 减少梯度累积
        warmup_ratio=0.03,  # 使用比例而非固定步数
        logging_steps=20,
        eval_steps=100,
        save_steps=200,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        learning_rate=3e-4,  # A100可增大学习率
        fp16=False,  # A100用bf16
        bf16=True,  # A100支持bfloat16
        optim="paged_adamw_32bit",  # 32bit优化器
        max_grad_norm=0.5,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        logging_dir="./a100_logs",
        save_total_limit=5,  # 多保存几个检查点
        push_to_hub=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=4,  # A100可增加worker
        remove_unused_columns=False,
        group_by_length=True,  # 按长度分组，提高效率
    )
    
    # 5. 格式化函数
    def format_for_a100(example):
        """A100优化的格式化"""
        text = f"""<|im_start|>system
你是一个专业的中文命名实体识别助手。<|im_end|>
<|im_start|>user
请识别以下文本中的实体：

{example['input']}

格式：{{实体类型: 实体}}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        return text
    
    # 6. 创建训练器
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        max_seq_length=2048,  # A100可处理更长序列
        packing=True,  # 开启packing提高效率
        formatting_func=format_for_a100,
        dataset_text_field="text",
    )
    
    # 7. 显示训练配置
    print("\n🚀 A100训练配置:")
    print(f"   Batch size: {training_args.per_device_train_batch_size}")
    print(f"   梯度累积: {training_args.gradient_accumulation_steps}")
    print(f"   有效批次: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"   学习率: {training_args.learning_rate}")
    print(f"   训练轮数: {training_args.num_train_epochs}")
        # print(f"   序列长度: {trainer.sequence_length}")
    print(f"   精度: {'bfloat16' if training_args.bf16 else 'float16'}")
    
    # 8. 开始训练
    print("\n⏳ 开始训练...")
    train_result = trainer.train()
    
    # 9. 保存模型
    print("\n💾 保存模型...")
    trainer.save_model("./qwen2.5-7b-ner-qlora-a100-final")
    tokenizer.save_pretrained("./qwen2.5-7b-ner-qlora-a100-final")
    
    # 保存训练指标
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    
    print("✅ 训练完成！")
    print(f"📁 模型保存到: ./qwen2.5-7b-ner-qlora-a100-final")
    
    return trainer

def quick_test_a100():
    """快速测试A100性能"""
    print("\n⚡ A100快速性能测试...")
    
    # 测试矩阵运算速度
    device = torch.device("cuda")
    
    # 大矩阵乘法测试
    size = 8192
    a = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    b = torch.randn(size, size, device=device, dtype=torch.bfloat16)
    
    torch.cuda.synchronize()
    import time
    start = time.time()
    
    for _ in range(10):
        c = torch.matmul(a, b)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"✅ A100性能测试完成")
    print(f"   8192x8192矩阵10次乘法: {elapsed:.3f}秒")
    print(f"   平均每次: {elapsed/10:.3f}秒")
    
    # 显存测试
    print(f"\n💾 显存测试:")
    print(f"   总显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"   已用显存: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"   缓存显存: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test_a100()
    else:
        # 运行性能测试
        quick_test_a100()
        
        # 开始训练
        trainer = train_on_a100()
        
        if trainer is not None:
            # 训练后快速评估
            print("\n📊 训练完成，开始评估...")
            os.system("python inference_a100.py")