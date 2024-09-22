import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import json
from tqdm import tqdm
import re  # 用于字符串清理

# 加载模型
model = Qwen2VLForConditionalGeneration.from_pretrained(
    "/root/autodl-tmp/Qwen/Qwen2-VL-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# 加载处理器
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/Qwen/Qwen2-VL-7B-Instruct")


def clean_generated_text(text):
    """清理生成的文本，去除不必要的系统和用户消息，只保留assistant后的内容。"""
    # 使用正则表达式移除'system', 'user'等无关部分，只保留'assistant'后面的内容
    match = re.search(r"assistant\s*\n(.*)", text, re.S)
    if match:
        return match.group(1).strip()
    return text.strip()


def process_image_and_generate_description(image_path):
    """处理单个图像文件并返回生成的文本描述。"""
    try:
        # 打开图片
        image = Image.open(image_path)

        # 设置用户消息，包括图像和要生成的文本内容
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,  # 使用打开的图片对象
                    },
                    {"type": "text",
                     "text": "请帮忙分析这张图片，可以帮助模型提高对图像内容的理解，包括对象识别、场景分析和整体内容，用英文回答，简洁，切勿捏造。（每一句话十个单词内总结,每句话一个场景）\n"
                             "输出格式参考如下（每句话加上序号）：\n"
                             "1.。。。。\n"
                             "2.。。。。\n"
                             "3.。。。。\n"},
                ],
            }
        ]

        # 准备文本输入
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # 处理视觉输入信息
        image_inputs, video_inputs = process_vision_info(messages)

        # 准备模型输入，包括文本和图像
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")  # 将输入移动到 GPU

        # 执行推理生成描述
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=512)

        # 处理生成的输出
        generated_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # 清理生成的文本
        clean_text = clean_generated_text(generated_text)

        # 返回清理后的生成文本描述
        return clean_text

    except Exception as e:
        print(f"Failed to process image {image_path} with error: {str(e)}")
        return None


def process_jsonl_file(input_jsonl_path, output_jsonl_path):
    """逐个处理JSONL文件中的每个条目并保存更新后的条目。"""
    # 计算总的条目数以在进度条中显示
    with open(input_jsonl_path, 'r', encoding='utf-8') as infile:
        total_entries = sum(1 for _ in infile)

    processed_ids = set()  # 用于跟踪已处理的 id

    with open(input_jsonl_path, 'r', encoding='utf-8') as infile, open(output_jsonl_path, 'w',
                                                                       encoding='utf-8') as outfile:
        for line in tqdm(infile, total=total_entries, desc="Processing entries", unit="entry"):
            entry = json.loads(line.strip())
            entry_id = entry['id']
            if entry_id in processed_ids:
                continue  # 跳过已处理的条目
            processed_ids.add(entry_id)

            image_path = entry['images'][0]  # 假设每个 entry 只有一个图片路径
            result = process_image_and_generate_description(image_path)

            if result is not None:
                entry['text'] = f"<__dj__image>\n{result}"
                json.dump(entry, outfile, ensure_ascii=False)
                outfile.write('\n')


# 示例调用
input_jsonl_path = "test.jsonl"  # 替换为你的输入 JSONL 文件路径
output_jsonl_path = "test-vl.jsonl"  # 替换为输出 JSONL 文件路径

process_jsonl_file(input_jsonl_path, output_jsonl_path)
