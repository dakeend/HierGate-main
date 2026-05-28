import os
import re
import glob

def extract_base_name(filename):
    """
    从文件名提取基础名称
    1a5eA.txt -> 1a5e
    1a5eA_L37S.txt -> 1a5e
    """
    # 去除扩展名
    base = os.path.splitext(filename)[0]
    
    # 如果包含下划线，取第一个下划线前的部分
    if '_' in base:
        base = base.split('_')[0]
    
    # 去掉最后一个字符（通常是链标识符如A）
    if len(base) > 4:  # 确保不是太短的名称
        base = base[:-1]
    
    return base

def generate_pssm_from_folder(input_folder, output_folder, blast_db_path):
    """
    从文件夹中读取所有txt文件，生成PSSM特征
    
    Args:
        input_folder: 包含txt序列文件的文件夹路径
        output_folder: PSSM文件输出文件夹路径  
        blast_db_path: BLAST数据库路径
    """
    
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取所有txt文件
    txt_files = glob.glob(os.path.join(input_folder, "*.txt"))
    
    if not txt_files:
        print("未找到任何txt文件")
        return
    
    print(f"找到 {len(txt_files)} 个txt文件")
    
    # 临时fasta文件路径
    temp_fasta = os.path.join(input_folder, "Temporary.fasta")
    
    for txt_file in txt_files:
        try:
            # 获取文件名（不含路径）
            filename = os.path.basename(txt_file)
            print(f"正在处理: {filename}")
            
            # 读取序列
            with open(txt_file, 'r') as f:
                sequence = f.read().strip()
            
            # 提取基础名称作为FASTA header
            base_name = extract_base_name(filename)
            
            # 生成临时FASTA文件
            with open(temp_fasta, 'w') as f:
                f.write(f">{base_name}\n")
                f.write(sequence)
            
            # 生成输出PSSM文件路径
            output_pssm = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.pssm")
            
            # 构建psiblast命令
            # psiblast_cmd = (
            #     f'psiblast -query "{temp_fasta}" '
            #     f'-db "{blast_db_path}" '
            #     f'-evalue 0.001 -num_iterations 3 '
            #     f'-out_ascii_pssm "{output_pssm}"'
            # )
            psiblast_cmd = (
                f'psiblast -query "{temp_fasta}" '
                f'-db "{blast_db_path}" '
                f'-evalue 0.001  -num_iterations 3 '
                f'-out_ascii_pssm "{output_pssm}"'
            )
            # psiblast_cmd = (
            #     f'psiblast -query "{temp_fasta}" '
            #     f'-db "{blast_db_path}" '
            #     f'-evalue 10 -num_iterations 3 '
            #     f'-out_ascii_pssm "{output_pssm}"'
            # )
            
            # 执行psiblast
            result = os.system(psiblast_cmd)
            
            if result == 0:
                print(f"  ✓ {filename} -> {os.path.basename(output_pssm)}")
            else:
                print(f"  ✗ {filename} 处理失败")
                
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
    
    # 清理临时文件
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
    
    print("所有PSSM矩阵构建完成")

# 使用示例
if __name__ == "__main__":
    input_folder = "data/xulie/Tp53_test"
    output_folder = "data/pssm/Tp53_test"
    blast_db_path = "databases/swissprot"
    generate_pssm_from_folder(input_folder, output_folder, blast_db_path)
