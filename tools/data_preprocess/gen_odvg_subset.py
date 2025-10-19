import json
import os

def create_subset():
    # 输入和输出文件路径
    input_path = 'data/grounding_data/v3det/annotations/v3det_2023_v1_train_od.json'
    output_path = 'data/grounding_data/v3det/annotations/v3det_2023_v1_train_od.subset.json'
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 读取前100行
    print(f"正在读取文件: {input_path}")
    subset_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只读取前100行
                break
            instance = json.loads(line.strip())
            subset_data.append(instance)
    
    # 保存子集
    print(f"正在保存子集到: {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for instance in subset_data:
            f.write(json.dumps(instance, ensure_ascii=False) + '\n')
    
    print(f"处理完成！")
    print(f"子集包含 {len(subset_data)} 个实例")

if __name__ == '__main__':
    create_subset()
