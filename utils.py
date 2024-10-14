import requests
import json
from config import KB_URL

def kb_chat(
    query: str, 
    kb_name: str = "Chem", 
    top_k: int = 1, 
    score_threshold: float = 0.5, 
    prompt_name: str = 'default', 
    return_direct: str = True) -> str:
    
    url = KB_URL
    body = {
        "query": query,
        "mode": "local_kb",
        "kb_name": kb_name,
        "top_k": top_k,
        "score_threshold": score_threshold,
        "stream": False,
        "model": "qwen-turbo",
        "temperature": 0.0,
        "prompt_name": prompt_name,
        "return_direct": return_direct
    }

    headers = {
        'Content-Type': 'application/json'
    }

    resp = requests.post(url, headers=headers, json=body)
    resp_str = resp.json()
    resp_json = json.loads(resp_str)
    # print(resp_json)
    
    return resp_json['docs'][0]


#TODO 增加添加/删除文件的接口


if __name__ == '__main__':
    query = '下面是第七次全国人口普查中部分省份的人口数，请你将人口数按从多到少的顺序排列。 广东省：126012510人　　河南省：99365519人 山东省：101527453人　　四川省：83674866人'
    print(kb_chat(query=query, kb_name='problem2'))
