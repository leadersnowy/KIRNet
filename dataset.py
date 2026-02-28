import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from model import indicators, TIME_BINS
# =====================================================
# 医学先验知识：药物标准剂量映射（从“不同治疗方案的药物标准剂量表”提取）
# 键格式：药物名称（中文） → 标准剂量数值（单位统一为 mg/m² 或 mg 平剂量）
# =====================================================
STANDARD_DOSES = {
    # 化疗药物
    "长春瑞滨": 25,  # mg/m²
    "顺铂": 75,  # mg/m²
    "卡铂": 5,  # AUC=5（这里简化为数值5，实际计算时需特殊处理，但本项目暂不区分）
    "紫杉醇": 160,  # 135-175 的中间值，取约160
    "多西他赛": 70,  # 60-75 中间值
    "吉西他滨": 1125,  # 1000-1250 中间值
    "培美曲塞": 500,  # mg/m²（非鳞）
    "奈达铂": 100,  # mg/m²
    "紫杉醇脂质体": 160,  # 同紫杉醇
    "白蛋白结合型紫杉醇": 100,  # mg/m²（常见剂量）
    # 免疫治疗药物（均为固定剂量 mg）
    "帕博利珠单抗": 200,
    "替雷利珠单抗": 200,
    "卡瑞利珠单抗": 200,
    "信迪利单抗": 200,
    "纳武利尤单抗": 240,  # 常用3mg/kg，假设80kg患者约240mg
    "阿替利珠单抗": 1200,
    "度伐利尤单抗": 800,  # 10mg/kg 假设80kg
    # 靶向治疗药物（每日口服 mg）
    "吉非替尼": 250,
    "厄洛替尼": 150,
    "埃克替尼": 125,
    "达可替尼": 45,
    "阿法替尼": 40,
    "奥希替尼": 80,
    "克唑替尼": 250,
    "阿米替尼": 600,
    "塞瑞替尼": 450,
    # 抗血管生成
    "贝伐珠单抗": 11.25,  # 7.5~15 中间值
    "血管内皮抑制蛋白": 7.5,  # mg/kg 假设值
    "帕博利珠单抗": 200,  # 重复避免冲突
    "白蛋白紫杉醇": 100,  # 同上
}

# 治疗类型到是否为药物治疗的映射
DRUG_TREATMENTS = ['化疗', '免疫治疗', '靶向治疗']
# =====================================================
# Dataset 定义
# =====================================================
class TBRDataset(Dataset):
    def __init__(self, excel_path):
        xl = pd.ExcelFile(excel_path)
        sheet_names = xl.sheet_names

        # 1. 治疗前生理指标
        df_pre = pd.read_excel(excel_path, sheet_name="治疗前", header=1)
        col_mapping = {
            "BMI": "BMI",
            "收缩压": "收缩压（mmHg）",
            "舒张压": "舒张压（mmHg）",
            "葡萄糖": "葡萄糖",
            "胆固醇": "胆固醇",
            "LDL-C": "低密度脂蛋白胆固醇",
            "HDL-C": "高密度脂蛋白胆固醇",
            "D-二聚体": "D-二聚体（ng/ml）",
            "血小板": "血小板（10^9/L）",
            "中性粒细胞":"中性粒细胞（10^9/L）",
            "淋巴细胞": "淋巴细胞（10^9/L）",
            "NLR（中性粒细胞/淋巴细胞）": "NLR（中性粒细胞/淋巴细胞）",
            "神经元特异性烯醇化酶": "神经元特异性烯醇化酶（ng/ml）",
            "鳞状细胞癌相关抗原": "鳞状细胞癌相关抗原（ng/ml）",
            "胃泌素释放肽前体": "胃泌素释放肽前体（pg/ml）",
            "细胞角蛋白19片段": "细胞角蛋白19片段（ng/ml）",
            "CRP": "CRP（mg/L）",
            "降钙素原": "降钙素原（ng/ml）",
            "IL-6": "IL-6（pg/ml）",
            "BNP": "BNP（pg/ml）",
            "高敏肌钙蛋白T": "高敏肌钙蛋白T（ug/L）",
            "肾小球滤过率": "肾小球滤过率（ml/min/1.73m^2）",
            "肌酐": "肌酐（µmol/L）",
            "脂蛋白a": "脂蛋白a（mg/l）",
            "游离三碘甲状腺原氨酸": "游离三碘甲状腺原氨酸",
            "促甲状腺激素": "促甲状腺激素",
            "游离甲状腺素": "游离甲状腺素",
            "甘油三酯": "甘油三酯"
        }
        df_pre = df_pre.rename(columns=lambda x: str(x).strip())

        # 关键修改：处理 <0.5、>10 这类字符串
        def parse_limit_value(val):
            if pd.isna(val):
                return np.nan
            s = str(val).strip()
            # 去除可能的两端空格和单位（如 "<0.5 "）
            s = s.split()[0] if ' ' in s else s  # 只取数字部分
            if s.startswith(('<', '＜', '≤', '<=')):
                # <0.5、＜0.5、≤0.003 → 用边界值 0.5、0.003
                try:
                    return float(s[1:])
                except:
                    return np.nan
            elif s.startswith(('>', '＞', '≥', '>=')):
                # >10、＞10、≥20 → 用边界值 10、20
                try:
                    return float(s[1:])
                except:
                    return np.nan
            else:
                try:
                    return float(s)
                except:
                    return np.nan

        # 先取出需要的列
        selected_cols = [col_mapping[ind] for ind in indicators]
        df_selected = df_pre[selected_cols].copy()

        # 对所有单元格应用解析函数
        df_selected = df_selected.map(parse_limit_value)

        # 转为 float32（缺失值仍为 NaN，后面聚合时会自动处理）
        df_pre_num = df_selected.astype(np.float32)
        X_pre = df_pre_num.values

        # 2. 治疗中指标聚合
        patient_sns = df_pre["patient_SN"].tolist()
        X_mid_agg = np.zeros_like(X_pre)
        for i, psn in enumerate(patient_sns):
            sheet = f"治疗中{psn}"
            if sheet not in sheet_names:
                continue
            df_mid_raw = pd.read_excel(excel_path, sheet_name=sheet, header=None)
            labels = df_mid_raw.iloc[1:, 0].tolist()  # 从row2开始的行标签
            dates = df_mid_raw.iloc[0, 1:].tolist()   # row1的日期，从col2开始
            data = df_mid_raw.iloc[1:, 1:]           # 值矩阵
            df_mid = pd.DataFrame(data.values, index=labels, columns=dates)
            df_mid = df_mid.apply(pd.to_numeric, errors="coerce")
            try:
                selected = df_mid.loc[indicators]  # (len(indicators), num_dates)
                agg = np.nanmean(selected.values, axis=1)  # (len(indicators),)
                X_mid_agg[i] = agg
            except KeyError:
                pass  # 如果缺少指标，保持0

        # 3. 拼接治疗前 + 治疗中
        self.X = X_pre + X_mid_agg
        self.X = np.nan_to_num(self.X, nan=0.0)
        # 对生理指标做标准化
        mean = np.nanmean(self.X, axis=0)
        std = np.nanstd(self.X, axis=0) + 1e-6
        self.X = (self.X - mean) / std

        assert self.X.shape[1] == len(indicators)

        # 4. 目标 TBR
        df_post = pd.read_excel(excel_path, sheet_name="治疗后", header=0)
        self.y = df_post["胸主动脉TBR值"].values.astype(np.float32)
        self.y = np.nan_to_num(self.y, nan=0.0)

        # 5. 预测时间区间（修正版）
        def month_to_bin(m):
            if np.isnan(m):
                return 0
            m = float(m)
            if 3 <= m < 9:
                return 0
            elif 9 <= m < 15:
                return 1
            elif 15 <= m < 21:
                return 2
            elif 21 <= m < 27:
                return 3
            else:
                return 0

        # 直接使用 pandas 解析的 Timestamp
        post_dates = pd.to_datetime(df_post["影像时间"], errors='coerce')  # 转为 Timestamp
        pre_dates = pd.to_datetime(df_pre["PETCT影像"], errors='coerce')  # 治疗前影像日期

        # 计算相差月数（除以平均一个月30.4375天）
        months = [(p - pre).days / 30.4375 if pd.notna(p) and pd.notna(pre) else np.nan
                  for p, pre in zip(post_dates, pre_dates)]

        self.time_idx = np.array([month_to_bin(m) for m in months], dtype=np.int64)

        # 6. 治疗事件解析（关键修改：解析实际剂量并计算 dose_ratio）
        self.last_treat_types, self.delta_t, self.treat_events, self.current_time = [], [], [], []
        origin = np.datetime64("1899-12-30")
        treat_types = ['化疗', '放疗', '手术治疗', '免疫治疗', '靶向治疗']
        treat_row_starts = {'化疗': 31, '放疗': 32, '手术治疗': 33, '免疫治疗': 34, '靶向治疗': 35}
        for i, psn in enumerate(patient_sns):
            sheet = f"治疗中{psn}"
            if sheet not in sheet_names:
                self.last_treat_types.append("化疗")
                self.delta_t.append(0.0)
                self.current_time.append(0.0)
                self.treat_events.append([])
                continue
            df_t = pd.read_excel(
                excel_path,
                sheet_name=sheet,
                header=None,
                nrows=36  # 强制读取前 36 行，即使后面为空也不会被裁剪
            )
            dates_excel = pd.to_numeric(df_t.iloc[0, 1:], errors="coerce")
            dates = np.array([origin + np.timedelta64(int(x), "D") if not np.isnan(x) else np.datetime64("NaT") for x in dates_excel])
            events = []
            last_type = '化疗'
            valid_dates = dates[~pd.isna(dates)]
            min_date = valid_dates.min() if len(valid_dates) > 0 else np.datetime64("NaT")
            max_date = valid_dates.max() if len(valid_dates) > 0 else np.datetime64("NaT")
            for t_type in treat_types:
                row_idx = treat_row_starts[t_type]
                treat_row = df_t.iloc[row_idx, 1:]
                for d_idx, cell in enumerate(treat_row):
                    if pd.isna(cell) or pd.isna(dates[d_idx]):
                        continue
                    cell_str = str(cell).strip()
                    if cell_str == '' or cell_str.lower() == 'nan':
                        continue
                    # 处理多个药物：按 '+' 分割
                    sub_treats = cell_str.split('+')
                    for sub_cell in sub_treats:
                        sub_cell = sub_cell.strip()
                        if not sub_cell:
                            continue
                        # 解析剂量字符串，如 "白蛋白紫杉醇400mgd1", "卡铂500mgd1", "帕博利珠单抗200mgd1"
                        dose_ratio = 1.0
                        is_drug = t_type in DRUG_TREATMENTS
                        if is_drug:
                            # 提取药物名称和剂量数字
                            import re
                            # 匹配数字 + 可选单位
                            match = re.search(r'(\d+(\.\d+)?)\s*(mg|m²|mgd|mg/m²)?', sub_cell, re.IGNORECASE)
                            drug_name = sub_cell
                            actual_dose = None
                            if match:
                                actual_dose = float(match.group(1))
                                # 药物名称取数字前的部分
                                drug_name = sub_cell[:match.start()].strip()
                            # 如果没匹配到药物名，尝试常见药物关键词
                            if not drug_name:
                                for drug in STANDARD_DOSES.keys():
                                    if drug in sub_cell:
                                        drug_name = drug
                                        break
                            if actual_dose is not None and drug_name in STANDARD_DOSES:
                                standard = STANDARD_DOSES[drug_name]
                                dose_ratio = actual_dose / standard
                            # else: 无法解析，保持 dose_ratio=1.0
                        rel_t = ((dates[d_idx] - min_date) / np.timedelta64(365, "D") if not np.isnat(min_date) else 0.0)
                        events.append({
                            "type": t_type,
                            "time": float(rel_t),
                            "dose_ratio": float(dose_ratio),
                            "is_drug": is_drug
                        })
                        last_type = t_type
            # 时间计算
            self.last_treat_types.append(last_type)
            post_date = pd.to_datetime(df_post.iloc[i]["影像时间"])
            if pd.isna(post_date):
                post_date = np.datetime64('NaT')
            else:
                post_date = post_date.to_numpy()  # 转为 np.datetime64
            delta = ((post_date - max_date) / np.timedelta64(365, "D") if not np.isnat(max_date) else 0.0)
            total = ((post_date - min_date) / np.timedelta64(365, "D") if not np.isnat(min_date) else 0.0)
            self.delta_t.append(float(delta))
            self.current_time.append(float(total))
            self.treat_events.append(events)

        self.delta_t = np.array(self.delta_t, dtype=np.float32)
        self.current_time = np.array(self.current_time, dtype=np.float32)

        # 确保 current_time 和 delta_t 不出现 NaN，用 0 填充
        self.delta_t = np.nan_to_num(self.delta_t, nan=0.0)
        self.current_time = np.nan_to_num(self.current_time, nan=0.0)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        TREAT_TYPE2IDX = {
            "化疗": 0,
            "放疗": 1,
            "手术治疗": 2,
            "免疫治疗": 3,
            "靶向治疗": 4
        }
        return {
            "phys_values": torch.tensor(self.X[idx], dtype=torch.float32),
            "target": torch.tensor(self.y[idx], dtype=torch.float32),
            "time_idx": torch.tensor(self.time_idx[idx], dtype=torch.long),
            # "last_treat_types": self.last_treat_types[idx],
            "last_treat_types": torch.tensor(
                TREAT_TYPE2IDX.get(self.last_treat_types[idx], 0),
                dtype=torch.long
            ),
            "delta_t": torch.tensor(self.delta_t[idx], dtype=torch.float32),
            "treat_events": self.treat_events[idx],
            "current_time": torch.tensor(self.current_time[idx], dtype=torch.float32)
            }

    def collate(self, batch):
        return {
            "phys_values": torch.stack([b["phys_values"] for b in batch]),
            "target": torch.stack([b["target"] for b in batch]),
            "time_idx": torch.stack([b["time_idx"] for b in batch]),
            # "last_treat_types": [b["last_treat_types"] for b in batch],
            "last_treat_types": torch.stack(
                [b["last_treat_types"] for b in batch]
            ),
            "delta_t": torch.stack([b["delta_t"] for b in batch]),
            "treat_events": [b["treat_events"] for b in batch],
            "current_time": torch.stack([b["current_time"] for b in batch])
        }
