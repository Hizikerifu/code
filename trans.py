import os
import xml.etree.ElementTree as ET

def convert_xml_to_yolo(xml_path, output_dir):
    # 读取XML文件
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # 获取图像尺寸
    size = root.find('size')
    img_width = float(size.find('width').text)
    img_height = float(size.find('height').text)
    
    # 准备输出文件
    xml_filename = os.path.basename(xml_path)
    txt_filename = xml_filename.replace('.xml', '.txt')
    txt_path = os.path.join(output_dir, txt_filename)
    
    # 打开输出文件
    with open(txt_path, 'w') as f:
        # 处理每个目标
        for obj in root.findall('object'):
            # 获取类别 (这里假设'fire'为类别0)
            class_name = obj.find('name').text
            class_id = 0 if class_name == 'fire' else -1
            
            # 获取边界框坐标
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # 转换为YOLO格式
            # YOLO格式: <class_id> <x_center> <y_center> <width> <height>
            x_center = (xmin + xmax) / (2.0 * img_width)
            y_center = (ymin + ymax) / (2.0 * img_height)
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height
            
            # 写入txt文件
            f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")

def main():
    # 设置输入和输出目录
    annotations_dir = 'datasets/Annotations'  # XML文件目录
    output_dir = 'datasets/labels'  # 输出的txt文件目录
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 遍历所有XML文件
    for xml_file in os.listdir(annotations_dir):
        if xml_file.endswith('.xml'):
            xml_path = os.path.join(annotations_dir, xml_file)
            convert_xml_to_yolo(xml_path, output_dir)
            print(f"Converted {xml_file} to YOLO format")

if __name__ == "__main__":
    main()