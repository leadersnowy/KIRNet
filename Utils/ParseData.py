import os
import time
import requests
from Bio import Entrez
import pandas as pd
from datetime import datetime

os.makedirs("PubMed_100_papers_3to9months", exist_ok=True)
# 1. 顶级期刊白名单（全部2025年IF>15 或权威指南库）
HIGH_IMPACT_JOURNALS = [
    "JACC: CardioOncology",
    "Cochrane Database of Systematic Reviews",
    "European Heart Journal",
    "Circulation",
    "Lancet Oncology",
    "Journal of Clinical Oncology",
    "JAMA Oncology",
    "Annals of Oncology",
    "European Journal of Cancer",
    "Cancer Discovery",
    "Nature Reviews Cardiology",
    "ESC Guidelines",                    # 额外加上 ESC 指南（经常出现在 Cochrane/EHJ）
    "Cardio-Oncology"                    # 部分新刊可能用这个名字
]
# 2. 癌症相关词（防止漏掉变体）
CANCER_TERMS = "(cancer OR neoplasm OR oncology OR tumor OR carcinoma OR malignancy)"
# 3. 五大治疗方案（含常用缩写和别名）
TREATMENT_TERMS = """
(chemotherapy OR radiotherapy OR radiation OR surgery OR surgical OR 
 immunotherapy OR "immune checkpoint" OR "PD-1" OR "PD-L1" OR ICI OR 
 "targeted therapy" OR TKI OR "HER2" OR "CDK4/6" OR PARP OR "endocrine therapy")
"""
# 4. 七大生理变量（必须至少命中2个）
METABOLIC_VARIABLES = """
(BMI OR "body mass index" OR "blood pressure" OR hypertension OR glucose OR 
 glycemia OR cholesterol OR "total cholesterol" OR LDL OR "LDL-C" OR HDL OR "HDL-C" OR 
 triglyceride OR triglycerides OR "lipid profile" OR dyslipidemia OR "metabolic syndrome")
"""
# 5. 强制锁定 3–9 个月时间窗
TIME_WINDOW_3TO9M = """
("3 month" OR "3 months" OR "6 month" OR "6 months" OR "9 month" OR "9 months" OR
 "3-6 month" OR "3-9 month" OR "90 day" OR "90 days" OR "180 day" OR "180 days" OR 
 "270 day" OR "270 days" OR "at 3 months" OR "at 6 months" OR "at 9 months" OR
 "3-month" OR "6-month" OR "9-month" OR "6-month follow-up" OR "6 month follow-up" OR
 quarterly OR "first 6 months" OR "first year" OR "within 12 months")
"""
# 6. 高证据等级文献类型
EVIDENCE_TYPE = "(meta-analysis[ptyp] OR systematic[sb] OR guideline[ptyp] OR review[ptyp] OR consensus[ptyp] OR practice guideline[ptyp])"
# 7. 年份限制
YEAR_RANGE = "(\"2020\"[PDAT] : \"2025\"[PDAT])"
# 终极查询
query = f"""
({" OR ".join([f'"{j}"[Journal]' for j in HIGH_IMPACT_JOURNALS])})
AND {CANCER_TERMS}
AND {TREATMENT_TERMS}
AND {METABOLIC_VARIABLES}
AND {TIME_WINDOW_3TO9M}
AND {YEAR_RANGE}
AND {EVIDENCE_TYPE}
"""

def search_pubmed():
    print(f"[{datetime.now()}] 开始搜索…")
    handle = Entrez.esearch(
        db="pubmed",
        term=query,
        retmax=500,          # 多拉一点，后面精选前 100
        sort="relevance",
        usehistory="y"
    )
    result = Entrez.read(handle)
    handle.close()
    total = int(result["Count"])
    webenv = result["WebEnv"]
    query_key = result["QueryKey"]
    print(f"PubMed 共命中 {total} 篇顶级文献")
    return webenv, query_key, total
def fetch_pmids(webenv, query_key, retmax=500):
    handle = Entrez.efetch(
        db="pubmed",
        retmode="xml",
        retstart=0,
        retmax=retmax,
        webenv=webenv,
        query_key=query_key
    )
    records = Entrez.read(handle)
    handle.close()
    pmids = [str(article["MedlineCitation"]["PMID"]) for article in records["PubmedArticle"]]
    return pmids

def get_pdf_url(rec):
    """从 XML 中提取 PMC 全文 PDF 直链（最快最准）"""
    if "PubmedData" in rec and "ArticleIdList" in rec["PubmedData"]:
        for aid in rec["PubmedData"]["ArticleIdList"]:
            if aid.attributes.get("IdType") == "pmc":
                pmc = str(aid)
                return f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmc}/pdf/"
    return None

def download(pdf_url, path):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(pdf_url, headers=headers, timeout=40)
        if r.status_code == 200 and len(r.content) > 80000:  # >80KB 基本是真PDF
            with open(path, "wb") as f:
                f.write(r.content)
            return True, len(r.content)
    except:
        pass
    return False, 0


# ===================== 主程序 =====================
if __name__ == "__main__":
    webenv, query_key, total = search_pubmed()
    pmids = fetch_pmids(webenv, query_key, retmax=500)

    # 批量获取详细信息（含 PDF 链接）
    print(f"[{datetime.now()}] 批量获取 300+ 篇的元数据+PDF链接…")
    handle = Entrez.efetch(db="pubmed", id=pmids, retmode="xml")
    records = Entrez.read(handle)["PubmedArticle"]
    handle.close()

    success_list = []
    count = 0

    for i, rec in enumerate(records):
        if count >= 100:
            break

        pmid = rec["MedlineCitation"]["PMID"]
        art = rec["MedlineCitation"]["Article"]
        title = art.get("ArticleTitle", "No title")
        journal = art.get("Journal", {}).get("Title", "")
        year = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {}).get("Year", "NA")

        pdf_url = get_pdf_url(rec)
        if not pdf_url:
            continue

        filename = f"PubMed_100_papers_3to9months/{pmid}_{year}_{journal[:40].replace(' ', '_')}.pdf"

        print(f"[{count + 1}/100] 下载中… {year} | {journal[:50]}")
        ok, size_kb = download(pdf_url, filename)

        if ok:
            count += 1
            success_list.append({
                "Rank": count,
                "PMID": pmid,
                "Year": year,
                "Journal": journal,
                "Title": title[:200],
                "PDF_Size_KB": round(size_kb / 1024, 1),
                "Local_Path": filename
            })
        time.sleep(0.8)  # NCBI 友好间隔

    # 保存清单
    df = pd.DataFrame(success_list)
    df.to_csv("PubMed_100_papers_3to9months_FINAL_LIST.csv", index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("大功告成！成功下载 100 篇「癌症治疗后 3–9 个月」顶级论文+指南")
    print("全部来自 JACC:CardioOncology、Cochrane、EHJ、Lancet Oncol 等")
    print("PDF 位置：PubMed_100_papers_3to9months/")
    print("清单位置：PubMed_100_papers_3to9months_FINAL_LIST.csv")
    print("=" * 80)