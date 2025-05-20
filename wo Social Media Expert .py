#coding=utf-8
import os
import random
import re
from openai import OpenAI
import pandas as pd
import stance_label
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

client = OpenAI(
    api_key="sk-CYrPc4NV7wy9WyaUgg55LJxcos0ONtJXGcpjxHiQPpEV3zi9",
    base_url="https://yunwu.ai/v1",
)
import time

def set_output(text):
    pattern = r"\*\*Analysis\*\*:\s*(.*?)\s*\*\*Stance Judgment"
    result = re.search(pattern, text, re.DOTALL)

    if result:
        content = result.group(1).strip()
    else:
        content = ""
    return content

def set_output_sl(text):
    # 正则表达式模式
    pattern = r"(.*?)\*\*Stance Judgment"  # 非贪婪匹配所有字符，包括换行符
    match = re.search(pattern, text, re.DOTALL)  # re.DOTALL 允许 . 匹配换行符

    if match:
        content = match.group(1).strip()
    else:
        content = ""

    return content

# @retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def get_openai_result(prompt):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # "llama3.1:405b",  #"llama-3.1-405B-instruct", gpt-3.5-turbo llama-3.1-70b
        messages=[{"role": "system", "content": 'You are the ultimate decision-maker of the opinion stance, adept at refining and summarizing content, and making your own decisions.'},{"role": "user", "content": prompt}],
        temperature=0,
        # top_p=1,
        # max_tokens=10000,
        # timeout=120
    )
    # print(response.choices[0].message.content.strip())
    return response.choices[0].message.content

if __name__ == "__main__":
    df=pd.read_excel('result/data.xlsx')
    df.loc[df['target'] == 'Atheism','sl']=stance_label.target1
    df.loc[df['target'] == 'Climate Change is a Real Concern', 'sl'] = stance_label.target2
    df.loc[df['target'] == 'Feminist Movement', 'sl'] = stance_label.target3
    df.loc[df['target'] == 'Hillary Clinton', 'sl'] = stance_label.target4
    df.loc[df['target'] == 'Legalization of Abortion', 'sl'] = stance_label.target5
    comment_list = df['text'].tolist()
    bg_list = df['bg'].tolist()
    sl_list = df['sl'].tolist()
    target_list = df['target'].tolist()
    label_list = df['label'].tolist()


    df1=pd.read_excel('result/sl2.xlsx')
    output_sl_list = df1['output'].apply(set_output_sl).tolist()

    df2=pd.read_excel('result/bgV2.xlsx')
    bgV2_list = df2['output'].apply(set_output).tolist()

    df3=pd.read_excel('result/am2.xlsx')
    am2_list = df3['output'].apply(set_output).tolist()
    for i in range(0,len(comment_list)):


        text = '''    
#Background#：{bg}

#Comment#:{comment}

Two experts have analyzed the stance of the comment from two perspectives: ① inferring background knowledge, and ② analyzing rhetorical devices. Please combine the analysis of the two experts, extract useful information, make your own analysis, and ultimately determine the stance of the comment on the target "{target}". Choose from options A, B, and C.
{sl}

#Knowledge reasoning expert#：
{bg_output}

#Pragmatic expert#：
{am_output}

#Answer format as follows#:
**Analysis**:
**Stance Judgment**:
                '''.strip()
        prompt = text.format(target=target_list[i],sl=sl_list[i], comment=comment_list[i],bg=bg_list[i],sl_output=output_sl_list[i],bg_output=bgV2_list[i],am_output=am2_list[i])
        # print(prompt)
        # break
        result = get_openai_result(prompt)
        # print(result)
        # break
        new_df = pd.DataFrame(
            {
                'text': [comment_list[i]],
                'target': [target_list[i]],
                'background': [bg_list[i]],
                'stance_label':[sl_list[i]],
                'label':[label_list[i]],
                'prompt': [prompt],
                'output': [result],
            })
        # 文件路径
        file_path = 'result/wosl.xlsx'

        # 检查文件是否存在
        if os.path.exists(file_path):
            # 读取现有数据
            existing_df = pd.read_excel(file_path)
            # 将新行追加到现有数据
            updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            # 如果文件不存在，直接使用新行作为初始数据
            updated_df = new_df

        # 将更新后的DataFrame写入Excel文件
        with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
            updated_df.to_excel(writer, sheet_name='Sheet1', index=False)
        count=i+1
        print('target:'+target_list[i]+'num:'+str(count))


