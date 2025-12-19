import pandas as pd
from datasets import Dataset
import os

def convert_sst2_to_hf_format(train_tsv_path, test_tsv_path, output_base_dir):
    """convert sst2-tsv to huggingface dataset format"""
    
    # 读取TSV文件
    train_df = pd.read_csv(train_tsv_path, sep='\t')
    test_df = pd.read_csv(test_tsv_path, sep='\t')
    
    # 由于calibration主要需要文本，我们只保留sentence列并重命名为text, 这样可以与wikitext2和c4的格式保持一致
    train_data = {'text': train_df['sentence'].tolist()}
    test_data = {'text': test_df['sentence'].tolist()}
    
    # 创建Dataset对象
    train_dataset = Dataset.from_dict(train_data)
    test_dataset = Dataset.from_dict(test_data)
    
    # 创建输出目录
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 保存为Hugging Face格式
    train_output_dir = os.path.join(output_base_dir, 'sst2_train')
    test_output_dir = os.path.join(output_base_dir, 'sst2_test')
    
    train_dataset.save_to_disk(train_output_dir)
    test_dataset.save_to_disk(test_output_dir)
    print(f"\n convertion finish!")
    print(f"train dataset in: {train_output_dir}")
    print(f"test dataset in: {test_output_dir}")
    
    # 验证文件创建
    print(f"\ntrain datafile: {os.listdir(train_output_dir)}")
    print(f"test datafile: {os.listdir(test_output_dir)}")

# 运行转换（只需执行一次）
convert_sst2_to_hf_format(
    train_tsv_path='/capsule/home/xiangyuxing/LOZO/LOZO/medium_models/data/k-shot-1k-test/SST-2/512-42/train.tsv',
    test_tsv_path='/capsule/home/xiangyuxing/LOZO/LOZO/medium_models/data/k-shot-1k-test/SST-2/512-42/test.tsv', 
    output_base_dir='/capsule/home/xiangyuxing/wanda/sst_sample'
)