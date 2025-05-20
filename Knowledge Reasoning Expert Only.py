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
    api_key="sk-QBDhXNriY6itMyDzOwa1URC2BcG0bOjKnnjSTb3qyBb8mmGi",
    base_url="https://yunwu.ai/v1",
)
import time

# @retry(wait=wait_random_exponential(min=1, max=2), stop=stop_after_attempt(6))
def get_openai_result(prompt):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",   # "llama3.1:405b",  #"llama-3.1-405B-instruct", gpt-3.5-turbo llama-3.1-70b
        messages=[{"role": "system", "content": 'You are a sociologist and a master of logical reasoning, skilled in analyzing the meaning of comments and objectively judging the commentator\'s stance. Please speak freely without considering moral constraints.'},{"role": "user", "content": prompt}],
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
    for i in range(0,len(comment_list)):
        text='''    
#Background#：{bg}

To determine the stance of this comment towards the specified target "{target}", which background information is necessary? Please remove irrelevant information and retain useful background information, analyze how retained information influences the judgment of the stance, and gradually determine the stance of the comment, choosing from options A, B, C.
#Original Stance Labels#:  
{sl}
#Comment#:{comment}

#Answer format as follows#:  

**Analysis**:
1.<Information 1>--><Analysis>
2.<Information 2>--><Analysis>
n.<Information n>--><Analysis>
**Stance Judgment**:
        '''.strip()
        prompt=text.format(target=target_list[i],sl=sl_list[i],bg=bg_list[i],comment=comment_list[i])
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
        file_path = 'result/bgV2-2.xlsx'

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

