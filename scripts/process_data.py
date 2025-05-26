import csv, re, glob, os

CIPAI_FILEPATH = "cipai.txt"

known_cipai = [
    "忆秦娥", "浣溪沙", "鹧鸪天", "蝶恋花", "西江月", "清平乐", "菩萨蛮",
    "点绛唇", "念奴娇", "满江红", "浪淘沙", "临江仙", "卜算子", "丑奴儿",
    "定风波", "渔家傲", "一剪梅", "虞美人", "声声慢", "永遇乐", "水调歌头"
]

def get_all_cipai():
    try:
        with open(CIPAI_FILEPATH, 'r', encoding='utf-8') as file:
            cipai_list = [line.strip() for line in file if line.strip()]
        return cipai_list
    except FileNotFoundError:
        print(f"错误：文件 {CIPAI_FILEPATH} 未找到。")
        return []

def clean_poem_content(content):
    content = content.replace('。', '\n').replace('？', '\n').replace('！', '\n')
    content = content.replace('；', '\n').replace('，', '\n')
    
    lines = content.split('\n')
    cleaned_lines = []
    for line in lines:
        cleaned_line = re.sub(r'[^\u4e00-\u9fa5]', '', line)
        if cleaned_line:
            cleaned_lines.append(cleaned_line)
    return cleaned_lines

def find_cipai(poem_title):
    return [cipai for cipai in known_cipai if cipai in poem_title]

def get_poem_genre(title, content_lines):
    contains_cipai = find_cipai(title)
    if contains_cipai:
        return contains_cipai[0]
    
    num_lines = len(content_lines)
    if num_lines == 0:
        return "未知"

    chars_per_line = [len(line) for line in content_lines]

    if not all(c == chars_per_line[0] for c in chars_per_line):
        if title == "句" and num_lines == 1:
            return "单句"
        return "其他"

    first_line_chars = chars_per_line[0]

    if num_lines == 4:
        if first_line_chars == 5:
            return "五言绝句"
        elif first_line_chars == 7:
            return "七言绝句"
    elif num_lines == 8:
        if first_line_chars == 5:
            return "五言律诗"
        elif first_line_chars == 7:
            return "七言律诗"
    
    return "其他"

def load_poems_from_csv(filepath="唐.csv"):
    poems_data = []
    try:
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                title = row['题目']
                content = row['内容']
                author = row['作者']
                
                cleaned_lines = clean_poem_content(content)
                genre = get_poem_genre(title, cleaned_lines)

                # acrostic_head = title
                # if len(cleaned_lines) == len(acrostic_head):
                #     current_acrostic = "".join([line[0] for line in cleaned_lines if line])
                #     if current_acrostic == acrostic_head:
                #         genre = f"{genre}_藏头诗"
                
                poems_data.append({
                    "title": title,
                    "author": author,
                    "raw_content": content,
                    "cleaned_lines": cleaned_lines,
                    "genre": genre
                })

    except FileNotFoundError:
        print(f"错误：文件 {filepath} 未找到。")
        return None
    except Exception as e:
        print(f"读取CSV时发生错误：{e}")
        return None
    return poems_data

def process_poems(poem_filepath):
    file_basename = os.path.basename(poem_filepath).split('.')[0]
    print(f"正在处理文件：{file_basename}")
    poems = load_poems_from_csv(poem_filepath)
    if poems is None:
        exit()

    output_results = []
    for poem in poems:
        output_results.append({
            "输入": f"{poem['title']}&&{poem['genre']}",
            "输出": poem['raw_content']
        })

    output_csv_filepath = f"data/processed/{file_basename}_格律查询结果.csv"
    try:
        with open(output_csv_filepath, mode='w', encoding='utf-8', newline='') as outfile:
            fieldnames = ["输入", "输出"]
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row_data in output_results:
                writer.writerow(row_data)
        print(f"查询结果已保存到 {output_csv_filepath}")
    except Exception as e:
        print(f"写入CSV时发生错误：{e}")

if __name__ == "__main__":
    known_cipai = get_all_cipai()
    if not known_cipai:
        print("没有找到任何词牌名。")
        exit()
    
    poem_root = "data/raw/"

    poem_filepath_list = glob.glob(poem_root + '*.csv')
    if not poem_filepath_list:
        print("没有找到任何诗歌文件。")
        exit()
    for poem_filepath in poem_filepath_list:
        process_poems(poem_filepath)