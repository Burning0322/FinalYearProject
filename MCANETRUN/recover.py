import os
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from selenium.common.exceptions import TimeoutException, WebDriverException
import pandas as pd
import time
import datetime

# 设置 requests 重试策略
session = requests.Session()
retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount('https://', HTTPAdapter(max_retries=retries))

# 初始化列表
PubChem_CID = []
error_cid = []
molecular_weight = []
molecular_formula = []
count = 0
total = 0

with open('KIBA.txt', 'r') as f:
    lines = f.readlines()

data = []
for line in lines:
    parts = line.strip().split(' ', 4)
    if len(parts) == 5:
        compound_id, protein_name, smiles, rest = parts[0], parts[1], parts[2], parts[3] + ' ' + parts[4]
        sequence, label = rest.rsplit(' ', 1)
        data.append({
            'compound_id': compound_id,
            'protein_name': protein_name,
            'smiles': smiles,
            'sequence': sequence,
            'label': int(label)
        })

ligands = set(d['compound_id'] for d in data)
sort_ligands = sorted(ligands)
print(len(sort_ligands))

# 设置 ChromeDriver 路径
local_path = "/MCANETRUN/chromedriver-mac-arm64/chromedriver"
service = Service(local_path)

for query in sort_ligands:
    browser = None
    try:
        start = time.time()
        print(f"处理化合物: {query}")
        browser = webdriver.Chrome(service=service)

        # 获取 PubChem 页面
        browser.get(f'https://pubchem.ncbi.nlm.nih.gov/#query={query}')

        # 等待并点击搜索结果链接
        result_link = WebDriverWait(browser, 30).until(
            EC.presence_of_element_located((By.XPATH, "//a[@data-ga-action='result-link']"))
        )
        result_link.click()

        # 等待页面跳转到化合物页面
        WebDriverWait(browser, 20).until(
            EC.url_contains("pubchem.ncbi.nlm.nih.gov/compound")
        )

        # 获取 CID
        css_selector = "div.text-left.sm\\:table-cell.sm\\:p-2.pb-1.pl-2.sm\\:align-middle"
        cid_element = WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, css_selector))
        )
        cid_text = cid_element.text.strip()
        print(f"compound_id={query}, PubChem CID={cid_text}")
        PubChem_CID.append(cid_text)

        # 获取 Molecular Weight
        mw_element = WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[@class='break-words space-y-1']"))
        )
        mw_text = mw_element.text.strip()
        print(f"Molecular Weight: {mw_text}")
        molecular_weight.append(mw_text)

        # 获取 Molecular Formula
        span_elem = WebDriverWait(browser, 20).until(
            EC.presence_of_element_located((By.XPATH, "//ul[@class='list-none']//span"))
        )
        raw_text = span_elem.text
        formula = re.sub(r"\s+", "", raw_text)
        print(f"Molecular Formula: {formula}")
        molecular_formula.append(formula)

        end = time.time()
        elapsed_time = end - start
        total += elapsed_time
        count += 1
        print(f"处理 {query} 完成，耗时: {elapsed_time:.2f} 秒, 处理了：{count}个，共用了{total:.2f}秒")
        time.sleep(2)

    except (TimeoutException, WebDriverException) as e:
        print(f"处理 {query} 超时: {str(e)}")
        error_cid.append(query)
    except Exception as e:
        print(f"处理 {query} 失败: {str(e)}")
        error_cid.append(query)
    finally:
        if browser:
            browser.quit()

print(f"成功处理 {count} 个化合物, 共用了 {total:.2f} 秒")

# 保存结果到 CSV 文件（带时间戳避免覆盖）
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
results = {
    'Query': sort_ligands[:len(PubChem_CID)],  # 确保长度匹配
    'PubChem CID': PubChem_CID,
    'Molecular Weight': molecular_weight,
    'Molecular Formula': molecular_formula
}
df = pd.DataFrame(results)
output_file = f'compound_properties_{timestamp}.csv'
df.to_csv(output_file, index=False)
print(f"已将结果保存到 {output_file}")

# 打印错误列表
if error_cid:
    print("以下化合物处理失败：")
    for cid in error_cid:
        print(f"  - {cid}")
else:
    print("所有化合物处理成功！")