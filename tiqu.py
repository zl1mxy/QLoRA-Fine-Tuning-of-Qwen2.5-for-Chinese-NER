import pandas as pd
import os

def incremental_merge():
    """增量合并第二个分片到已有数据"""
    
    # 已有数据路径
    existing_path = r"C:\Users\DELL\Desktop\NER\ner_output\ner_data.parquet"
    new_shard_path = r"C:\Users\DELL\Desktop\NER\train-00006-of-00272-7e9ef661e9319ac9.parquet"
    output_path = r"C:\Users\DELL\Desktop\NER\merged_ner_incremental.parquet"
    
    print("增量合并...")
    
    # 1. 读取已有数据
    if os.path.exists(existing_path):
        existing_df = pd.read_parquet(existing_path)
        print(f"已有数据: {len(existing_df):,} 行")
    else:
        print(f"❌ 找不到已有数据: {existing_path}")
        existing_df = pd.DataFrame()
    
    # 2. 读取并提取新分片的NER数据
    if os.path.exists(new_shard_path):
        new_df = pd.read_parquet(new_shard_path)
        print(f"新分片: {len(new_df):,} 行")
        
        if 'task_name_in_eng' in new_df.columns:
            new_ner_df = new_df[new_df['task_name_in_eng'] == 'named_entity_recognition']
            print(f"新分片NER数据: {len(new_ner_df):,} 行")
            
            # 3. 合并
            merged_df = pd.concat([existing_df, new_ner_df], ignore_index=True)
            
            # 4. 保存
            merged_df.to_parquet(output_path, index=False)
            print(f"✅ 合并完成！总共 {len(merged_df):,} 行")
            print(f"保存到: {output_path}")
            
            # 简单统计
            if 'other' in merged_df.columns:
                print("\n来源分布:")
                for source, count in merged_df['other'].value_counts().items():
                    print(f"  {source}: {count:,} 行")
                    
            return merged_df
        else:
            print("新分片缺少 task_name_in_eng 列")
            return existing_df
    else:
        print(f"❌ 找不到新分片: {new_shard_path}")
        return existing_df

# 运行
incremental_merge()