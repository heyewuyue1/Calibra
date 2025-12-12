import re
import json
import sys

def parse_log_to_json(log_file, output_file="output.json"):
    pattern = re.compile(r"1th execution time of (\S+): ([\d.]+)s")
    result = {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                sql_file = match.group(1)
                time_cost = float(match.group(2))
                result[sql_file] = time_cost

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)

    print(f"解析完成！共提取 {len(result)} 条记录，已写入 {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python log2json.py your_log.log [output.json]")
    else:
        log_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) > 2 else "output.json"
        parse_log_to_json(log_file, output_file)