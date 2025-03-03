import os
import re
import shutil
import argparse
from pathlib import Path

def convert_obsidian_to_hugo(input_file, output_folder, attachment_folder="G:\\my note\\快晴\\附件"):
    """
    将Obsidian Markdown文件转换为Hugo兼容的格式
    
    参数:
    input_file: Obsidian Markdown文件路径
    output_folder: 输出文件夹
    attachment_folder: Obsidian附件文件夹
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    # 创建图片文件夹
    images_folder = os.path.join(output_folder, "images")
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)
    
    # 读取Markdown文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 提取文件名（不含扩展名）
    file_basename = os.path.basename(input_file).split('.')[0]
    
    # 正则表达式匹配Obsidian图片语法: ![[图片名称.png]]
    obsidian_image_pattern = r'!\[\[(.*?)\]\]'
    
    # 查找所有图片引用
    image_matches = re.finditer(obsidian_image_pattern, content)
    
    # 图片索引计数器
    image_counter = 1
    
    # 存储图片的映射关系 {原始名称: 新名称}
    image_mapping = {}
    
    # 处理图片引用
    for match in image_matches:
        original_image_name = match.group(1)
        
        # 在Obsidian附件文件夹中查找图片
        original_image_path = os.path.join(attachment_folder, original_image_name)
        
        # 检查图片是否存在
        if os.path.exists(original_image_path):
            # 生成新的图片名称
            extension = os.path.splitext(original_image_name)[1]
            new_image_name = f"{file_basename}-image-{image_counter}{extension}"
            
            # 复制图片到新位置
            new_image_path = os.path.join(images_folder, new_image_name)
            shutil.copy2(original_image_path, new_image_path)
            
            # 更新映射
            image_mapping[original_image_name] = new_image_name
            
            # 增加计数器
            image_counter += 1
    
    # 修改脚本中的图片替换部分
    for original_name, new_name in image_mapping.items():
        # 替换Obsidian格式为标准Markdown格式，使用新文件名作为显示文本
        obsidian_pattern = f"!\\[\\[{re.escape(original_name)}\\]\\]"
        # 使用新文件名（不含扩展名）作为显示文本
        new_name_without_ext = os.path.splitext(new_name)[0]
        markdown_replacement = f"![{new_name_without_ext}](images/{new_name})"
        content = re.sub(obsidian_pattern, markdown_replacement, content)
    
    # 检查和修正front matter
    front_matter_pattern = r'^---\s*\n(.*?)\n---\s*\n'
    front_matter_match = re.search(front_matter_pattern, content, re.DOTALL)
    
    if front_matter_match:
        front_matter = front_matter_match.group(1)
        # 确保front matter包含Hugo所需的字段
        if 'title:' not in front_matter:
            title = file_basename.replace('-', ' ').title()
            front_matter += f'title: "{title}"\n'
        if 'date:' not in front_matter:
            from datetime import datetime
            current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
            front_matter += f'date: {current_date}\n'
        
        # 更新front matter
        content = re.sub(front_matter_pattern, f'---\n{front_matter}\n---\n\n', content, flags=re.DOTALL)
    else:
        # 如果没有front matter，添加一个
        title = file_basename.replace('-', ' ').title()
        from datetime import datetime
        current_date = datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00")
        front_matter = f'---\ntitle: "{title}"\ndate: {current_date}\ndraft: false\n---\n\n'
        content = front_matter + content
    
    # 创建index.md文件
    output_file = os.path.join(output_folder, "index.md")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"转换完成! 输出文件: {output_file}")
    print(f"处理了 {image_counter - 1} 个图片")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='将Obsidian Markdown文件转换为Hugo兼容格式')
    parser.add_argument('input_file', help='输入的Obsidian Markdown文件路径')
    parser.add_argument('output_folder', help='输出文件夹路径')
    parser.add_argument('--attachment', default="G:\\my note\\快晴\\附件", help='Obsidian附件文件夹路径')
    
    args = parser.parse_args()
    
    convert_obsidian_to_hugo(args.input_file, args.output_folder, args.attachment)