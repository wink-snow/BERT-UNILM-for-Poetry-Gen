import csv

FEILD_NAMES = ["输入", "输出"]
SPLIT_RATE = 0.8

def load_poems_from_csv(filepath: str, mode: str = "train") -> list[dict]:
    poems = []
    try: 
        with open(filepath, mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)

            for row in reader:
                input = row[FEILD_NAMES[0]]
                output = row[FEILD_NAMES[1]]
                poems.append({"input": input, "output": output})

    except FileNotFoundError:
        print(f"错误：文件 {filepath} 未找到。")
        return []
    except KeyError as e:
        print(f"错误：CSV 文件缺少字段 {e}。")
        return []
    except Exception as e:
        print(f"读取CSV时发生错误：{e}")
        return []
    
    split_index = int(len(poems) * SPLIT_RATE)
    train_poems = poems[:split_index]
    test_poems = poems[split_index:]

    if mode == "train":
        return train_poems
    elif mode == "val" or mode == "test":
        return test_poems
    else:
        print(f"错误：未知的模式 {mode}。")
        return []
    