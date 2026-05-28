import streamlit as st
import pandas as pd
import requests
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from scipy import stats

# 페이지 기본 설정 (와이드 레이아웃)
st.set_page_config(page_title="miRNA Expression Analyzer (Normal vs Cancer)", page_icon="🧬", layout="wide")

# CSS를 활용한 디자인 커스터마이징 (팬톤 2016 & 클라우드 댄서 테마)
st.markdown("""
<style>
    /* 전체 배경 (클라우드 댄서: PANTONE 11-4201) */
    .stApp {
        background-color: #F0EEEB;
    }
    
    /* 사이드바 배경 */
    div[data-testid="stSidebar"] {
        background-color: #F0EEEB;
        border-right: 2px solid #F7CAC9;
    }

    /* 상단 타이틀 라벨 숨기기 (label_visibility="collapsed"의 폴백) */
    div[data-testid="stRadio"] > div:first-child > label {
        display: none !important;
    }

    /* 사이드바 네비게이션 라디오 버튼 (셀 형태) */
    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        display: flex;
        align-items: center;
        padding: 15px 20px;
        background-color: #F7CAC9; /* 로즈쿼츠 */
        border-radius: 12px;
        margin-bottom: 12px;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        cursor: pointer;
    }
    
    /* 선택된 네비게이션 셀 (css :has 활용) */
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]),
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) {
        background-color: #92A8D1 !important; /* 세레니티 */
        box-shadow: 0 4px 6px rgba(146, 168, 209, 0.3);
    }
    
    /* 텍스트 색상 및 볼드 처리 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {
        color: #555555;
        margin: 0;
        font-weight: 600;
    }
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(div[aria-checked="true"]) p,
    div[data-testid="stRadio"] div[role="radiogroup"] > label:has(input:checked) p {
        color: white !important;
        font-weight: 800;
    }
    
    /* 네비게이션 라디오 버튼 기본 아이콘(원형) 숨기기 */
    div[data-testid="stRadio"] div[role="radiogroup"] > label > div:first-child {
        display: none !important;
    }

    /* 메인 타이틀 그라데이션 */
    .main-title {
        font-size: 3rem;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #F7CAC9, #92A8D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.2rem;
        color: #7a8b9e;
        margin-bottom: 2rem;
    }

    /* 메트릭 카드 (홈 화면 및 분석 화면) */
    .metric-card {
        background-color: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 10px rgba(146, 168, 209, 0.15); /* 세레니티 톤 그림자 */
        text-align: center;
        border: 2px solid #F7CAC9; /* 로즈쿼츠 테두리 */
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        border-color: #92A8D1; /* 호버 시 세레니티로 변경 */
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #92A8D1; /* 기본 수치 색상 세레니티 */
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    
    /* P-value 색상 */
    .p-value-significant {
        color: #F7CAC9 !important; /* 의미 있는 경우 로즈쿼츠로 강조 */
        font-weight: 800;
    }
    .p-value-nonsignificant {
        color: #94a3b8 !important;
    }

    /* 컨테이너 (차트 구획) 스타일링 */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 2px solid #92A8D1 !important; /* 차트 구획은 세레니티 테두리 */
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(146, 168, 209, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# --- 데이터 로드 로직 (고도화: 업로드 기능 추가) ---
@st.cache_data
def load_default_data():
    file_path = 'archive/GSE58606_data.csv'
    try:
        df = pd.read_csv(file_path)
        # 기본 데이터 컬럼명 정리
        cleaned_columns = []
        seen = {}
        for col in df.columns:
            clean_name = col.split(" : ")[1].strip() if " : " in col else col
            if clean_name in seen:
                seen[clean_name] += 1
                clean_name = f"{clean_name} ({seen[clean_name]})"
            else:
                seen[clean_name] = 1
            cleaned_columns.append(clean_name)
        df.columns = cleaned_columns
        
        # target_actual 값 변경
        if 'target_actual' in df.columns:
            df['target_actual'] = df['target_actual'].replace({
                'primary breast cancer': 'Cancer (Tumor)',
                'normal breast tissue': 'Normal'
            })
        return df
    except Exception as e:
        return None

# --- [추가] 실시간 외부 생물학 API 호출 함수 ---
@st.cache_data(show_spinner=False)  # 반복 호출 시 속도 저하를 막기 위한 캐싱
def fetch_mirna_insight(gene_name):
    """
    MyGene.info API를 사용하여 유전자의 타겟, 경로, 요약 정보를 실시간으로 가져옵니다.
    로컬 지식베이스 매칭을 우선 적용한 뒤, 없을 경우 외부 API로 복합 검색합니다.
    """
    # 1. 괄호나 공백 제거 전처리
    clean_name = gene_name.split('(')[0].strip()
    
    # 2. 로컬에 정의된 핵심 miRNA 지식 베이스 매칭 우선 적용
    miRNA_knowledge = {
        "hsa-miR-21": {
            "target": "PTEN, PDCD4", 
            "role": "Oncogenic (암 발생 촉진)", 
            "pathways": ["PI3K/Akt Signaling", "Apoptosis Inhibition", "TGF-beta Signaling"],
            "desc": "유방암에서 가장 흔하게 과발현되는 miRNA로, 종양 억제 유전자인 PTEN을 억제하여 세포 증식과 생존을 돕습니다."
        },
        "hsa-let-7a": {
            "target": "RAS, MYC, HMGA2", 
            "role": "Tumor Suppressor (종양 억제)", 
            "pathways": ["Cell Cycle Control (G1/S)", "Ras/MAPK Signaling", "Pluripotency Maintenance"],
            "desc": "세포의 분화와 성장을 조절하며, 발현이 감소할 경우 RAS/MYC 단백질이 증가하여 암세포의 무분별한 증식을 유도합니다."
        },
        "hsa-miR-155": {
            "target": "SOCS1, SHIP1, TP53INP1", 
            "role": "Oncogenic (면역 및 증식 조절)", 
            "pathways": ["NF-kappaB Signaling", "Inflammatory Response", "B-cell Development"],
            "desc": "염증 반응과 관련된 유전자들을 조절하며, 유방암세포의 침습성과 항암제 내성을 높이는 데 관여합니다."
        },
        "hsa-miR-10b": {
            "target": "HOXD10", 
            "role": "Metastasis-related (전이 관련)", 
            "pathways": ["EMT (Epithelial-Mesenchymal Transition)", "RhoC Signaling", "Migration & Invasion"],
            "desc": "암세포가 주변 조직으로 퍼져나가는 전이 과정을 촉진하는 핵심 인자로, HOXD10 억제를 통해 세포의 이동성을 높입니다."
        }
    }
    
    # 대소문자/하이픈 무시 부분 매칭 검사
    clean_lower = clean_name.lower().replace("-", "")
    for k, v in miRNA_knowledge.items():
        k_clean = k.lower().replace("-", "")
        # 대소문자/하이픈이 제거된 상태에서 접두사를 고려하여 부분 일치 확인
        if k_clean in clean_lower or clean_lower in k_clean:
            return v
        # 'let-7a' 같은 핵심 유전자명 부분이 포함되어도 매칭
        k_core = k_clean.replace("hsa", "")
        if k_core in clean_lower:
            return v

    # 3. 로컬 매칭 실패 시, 외부 API 호출용 이름 표준화
    base_name = clean_name
    for prefix in ['hsa-', 'mmu-', 'rno-']:
        if base_name.lower().startswith(prefix):
            base_name = base_name[len(prefix):]
            break
            
    import re
    # -3p, -5p 접미사 제거
    base_name = re.sub(r'-(3p|5p).*$', '', base_name, flags=re.IGNORECASE)
    
    # HGNC 표준 유전자명으로 매핑 (예: let-7a-2 -> MIRLET7A2, mir-21 -> MIR21)
    hgnc_symbol = base_name.upper().replace("-", "")
    if hgnc_symbol.startswith("LET"):
        hgnc_symbol = "MIR" + hgnc_symbol
    elif not hgnc_symbol.startswith("MIR"):
        hgnc_symbol = "MIR" + hgnc_symbol

    # 4. MyGene.info 복합 쿼리 구성 (Lucene OR 검색으로 매칭 확률 최대화)
    query = f'symbol:{hgnc_symbol} OR alias:{base_name} OR "{clean_name}"'
    url = "https://mygene.info/v3/query"
    params = {
        "q": query,
        "fields": "summary,pathway,name,alias",
        "species": "human"
    }
    
    # SSL 인증서 문제 경고 숨기기
    import urllib3
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    try:
        # verify=False 옵션을 주어 SSL 인증에 실패하는 특정 로컬 네트워크 환경 대응
        response = requests.get(url, params=params, timeout=5, verify=False)
        if response.status_code == 200:
            data = response.json()
            if data.get("hits"):
                hit = data["hits"][0]
                
                # 생물학적 요약 정보
                desc = hit.get("summary")
                if not desc:
                    desc = f"현재 공식 데이터베이스에 {hgnc_symbol} ({clean_name})에 대한 요약문이 등록되어 있지 않습니다. 아래 PubMed 링크를 참조하세요."
                
                # 핵심 타깃 (miRNA의 타겟 mRNA는 API로 즉시 가져오기 어려우므로 안내 메시지 제공)
                # API 응답에서 가능한 타깃 정보를 탐색합니다. 없을 경우 fallback 문구를 사용합니다.
                target_genes = None
                if data.get("hits"):
                    hit = data["hits"][0]
                    # MyGene.info 결과에서 가능한 키를 순차 탐색
                    for key in ["target", "symbol", "name"]:
                        if hit.get(key):
                            target_genes = hit[key]
                            break
                if not target_genes:
                    # 커스텀 fallback 문구
                    target_genes = "타깃 정보를 찾을 수 없습니다. PubMed에서 검색해 보세요."
                
                # 경로 (Pathways) 파싱
                pathways = []
                if "pathway" in hit:
                    pw_data = hit["pathway"]
                    for source in ["kegg", "reactome", "wikipathways"]:
                        if source in pw_data:
                            items = pw_data[source]
                            if isinstance(items, list):
                                pathways.extend([item.get("name") for item in items if item.get("name")])
                            elif isinstance(items, dict):
                                pathways.append(items.get("name"))
                
                if not pathways:
                    pathways = ["General miRNA Pathway", "Cellular Regulation"]
                
                # 역할 추정
                role = "Functional miRNA"
                if "suppressor" in desc.lower():
                    role = "Tumor Suppressor (종양 억제)"
                elif "oncogen" in desc.lower() or "promote" in desc.lower():
                    role = "Oncogenic (암 발생 촉진)"

                return {
                    "target": target_genes,
                    "role": role,
                    "pathways": pathways[:3],
                    "desc": desc
                }
    except Exception as e:
        import sys
        print(f"[fetch_mirna_insight Error] {e}", file=sys.stderr)
        pass

    # 최종 API 실패/미검색 시 기본 fallback 데이터
    return {
        "target": f"실시간 분석 중 ({hgnc_symbol} 미검색)",
        "role": "Bio-Data 전처리 중",
        "pathways": ["Signal Transduction", "Disease Pathway"],
        "desc": f"외부 NCBI/MyGene API에서 {clean_name} ({hgnc_symbol})의 실시간 Summary 정보를 가져오지 못했습니다. 질병 종류(Cancer/Normal)에 따른 통계 수치는 우측 플롯을 참고하시고, 상세 기전은 아래 PubMed 링크를 활용해 주세요."
    }

# 사이드바: 데이터 소스 및 메뉴 설정
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>🧭 Navigation</h2>", unsafe_allow_html=True)
    menu = st.radio("화면 이동", ["🏠 홈 (Overview)", "🔬 유전자 분석 (Analysis)"], label_visibility="collapsed")
    
    # 캐시 초기화 버튼 추가
    if st.button("🧹 API 캐시 초기화", use_container_width=True, help="외부 API 조회 결과 및 로컬 캐시를 강제로 비우고 화면을 새로고침합니다."):
        st.cache_data.clear()
        st.success("캐시가 성공적으로 비워졌습니다!")
        st.rerun()
        
    st.markdown("---")
    
    st.markdown("### 📂 데이터 소스 설정")
    data_source = st.radio(
        "분석할 데이터를 선택하세요",
        ["기본 샘플 데이터 (GSE58606)", "내 데이터 업로드 (CSV)"],
        help="직접 수집한 데이터를 분석하려면 '내 데이터 업로드'를 선택하세요."
    )
    
    uploaded_df = None
    if data_source == "내 데이터 업로드 (CSV)":
        st.info("⚠️ 분석을 위해 반드시 **CSV 형식(.csv)**의 파일을 업로드해 주세요.")
        uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.success("✅ 파일 업로드 성공!")
            except Exception as e:
                st.error(f"파일 오류: {e}")

# 최종 데이터 결정 및 컬럼 매핑
df = None
group_col = 'target_actual'
normal_label = 'Normal'
cancer_label = 'Cancer (Tumor)'

if data_source == "내 데이터 업로드 (CSV)" and uploaded_df is not None:
    df = uploaded_df
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 🎯 데이터 매핑 설정")
        # 그룹 정보가 있을 법한 컬럼 추천 (유니크 값이 적은 것)
        potential_group_cols = [col for col in df.columns if df[col].nunique() < 10]
        group_col_selected = st.selectbox("그룹(상태) 정보 컬럼", df.columns, index=df.columns.get_loc(potential_group_cols[0]) if potential_group_cols else 0)
        
        unique_values = df[group_col_selected].unique().tolist()
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            normal_label = st.selectbox("정상군(Normal) 값", unique_values, index=0)
        with col_m2:
            cancer_label = st.selectbox("암군(Cancer) 값", unique_values, index=min(1, len(unique_values)-1))
        
        # 분석용 컬럼 생성
        df['target_actual'] = df[group_col_selected].map({normal_label: 'Normal', cancer_label: 'Cancer (Tumor)'})
        group_col = 'target_actual'
else:
    df = load_default_data()

if df is not None:
    # 타겟 컬럼 및 유전자 목록 추출
    group_col = 'target_actual'
    non_gene_cols = ['target', 'target_actual', 'target_original']
    gene_cols = [col for col in df.columns if col not in non_gene_cols]
    
    # 데이터 수치화 (업로드 데이터의 경우 문자열로 인식될 수 있음)
    for col in gene_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    gene_cols_sorted = sorted(gene_cols)
    
    normal_count = len(df[df[group_col] == 'Normal'])
    cancer_count = len(df[df[group_col] == 'Cancer (Tumor)'])

    # ---------------------------------------------------------
    # 🏠 홈 화면 (Overview)
    # ---------------------------------------------------------
    if menu == "🏠 홈 (Overview)":
        st.markdown('<p class="main-title">🧬 miRNA Expression Analyzer</p>', unsafe_allow_html=True)
        st.markdown('<p class="sub-title">유방암 환자 조직 vs 정상 조직 발현량 차이 (GSE58606 데이터셋)</p>', unsafe_allow_html=True)
        
        st.info("**이 프로그램은 유방암 환자와 정상인의 miRNA 유전자 발현량 차이를 탐색하고 통계적으로 분석하기 위해 제작되었습니다.** 좌측 메뉴에서 '유전자 분석'을 클릭하여 본격적인 데이터 시각화를 확인하세요.")
        
        st.markdown("### 📊 데이터셋 요약 정보 (Dataset Overview)")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value" style="color: #F7CAC9;">{cancer_count} <span style="font-size:1.2rem; color:#cbd5e1;">vs</span> <span style="color: #92A8D1;">{normal_count}</span></div>
                <div class="metric-label">총 샘플 수 (Cancer vs Normal)</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-card">
                <div class="metric-value" style="color: #92A8D1;">{len(gene_cols):,}</div>
                <div class="metric-label">분석 가능 유전자(miRNA) 수</div>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 📋 분석 지침 가이드")
        st.markdown("""
        1. **데이터 소스**: 본 데이터는 GEO(Gene Expression Omnibus)의 GSE58606에서 추출되었습니다.
        2. **목적**: 질병(유방암) 상태에 따라 특이적으로 높거나 낮게 발현되는 바이오마커(miRNA)를 발굴합니다.
        3. **통계적 검증**: T-test를 기반으로 p-value를 계산하여 두 그룹 간의 차이가 통계적으로 유의미한지 직관적으로 제공합니다.
        """)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("### 🧬 생물학적 의의 및 바이오마커 역할 안내")
        
        # 3가지 핵심 역할 (진단, 예후/전이, 치료 표적) 카드형 레이아웃
        st.markdown("""
        <div style="background-color: #ffffff; border-radius: 12px; padding: 22px; border: 2px solid #92A8D1; box-shadow: 0 4px 12px rgba(146, 168, 209, 0.12); margin-bottom: 25px;">
            <h4 style="color: #334155; margin-top: 0; margin-bottom: 12px; font-weight: 700;">🔬 분석 결과(유의미한 차이)가 가지는 생물학적 의미</h4>
            <p style="font-size: 0.95rem; color: #555555; line-height: 1.65; margin-bottom: 22px;">
                유전자 분석 탭에서 T-test 결과 <b>P-value &lt; 0.05</b>로 확인되는 miRNA들은 정상 조직과 유방암 조직 간에 발현량의 통계적으로 유의미한 차이가 있음을 뜻하며, 이는 암의 생물학적 기전과 밀접하게 연관되어 <b>바이오마커(Biomarker)</b>로서 다양한 임상적 가치를 가집니다.
            </p>
            <div style="display: flex; gap: 15px; flex-wrap: wrap;">
                <div style="flex: 1; min-width: 250px; background-color: #f8fafc; border-left: 4px solid #F7CAC9; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <strong style="color: #e11d48; font-size: 0.95rem;">🔍 1. 조기 진단 바이오마커 (Diagnostic)</strong>
                    <p style="font-size: 0.85rem; color: #64748b; line-height: 1.55; margin-top: 8px; margin-bottom: 0;">
                        특정 miRNA가 정상 조직보다 유방암 조직에서 극적으로 과발현(또는 저발현)되는 패턴을 보여준다면, 이를 혈액이나 조직 샘플에서 검출하여 질병의 존재 여부를 초기에 판별하는 <b>진단 마커</b>로 개발할 수 있습니다.
                    </p>
                </div>
                <div style="flex: 1; min-width: 250px; background-color: #f8fafc; border-left: 4px solid #92A8D1; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <strong style="color: #2563eb; font-size: 0.95rem;">📈 2. 예후 및 전이 예측 (Prognostic)</strong>
                    <p style="font-size: 0.85rem; color: #64748b; line-height: 1.55; margin-top: 8px; margin-bottom: 0;">
                        암세포의 전이(EMT) 과정이나 혈관 신생을 유도하는 특정 pathway 관여 유전자(예: miR-10b의 HOXD10 억제 등)는 환자의 재발 위험성, 종양 악성도 및 5년 생존율을 예측하는 <b>예후 판정 지표</b>로 사용됩니다.
                    </p>
                </div>
                <div style="flex: 1; min-width: 250px; background-color: #f8fafc; border-left: 4px solid #F7CAC9; padding: 16px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.02);">
                    <strong style="color: #b45309; font-size: 0.95rem;">🎯 3. 새로운 치료 표적 (Therapeutic Target)</strong>
                    <p style="font-size: 0.85rem; color: #64748b; line-height: 1.55; margin-top: 8px; margin-bottom: 0;">
                        통계적으로 암군에서 과발현되는 유전자는 암 성장을 돕는 <b>온코미르(OncomiR)</b>로서 핵산 저해 치료제의 타겟이 되며, 저발현되는 유전자는 <b>종양 억제 유전자(Tumor Suppressor)</b>로서 기능 보완 치료법의 표적이 됩니다.
                    </p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # 과발현 vs 저발현 생물학적 기전 상세 카드
        st.markdown("""
        <div style="display: flex; gap: 20px; margin-bottom: 30px; flex-wrap: wrap;">
            <div style="flex: 1; min-width: 300px; background-color: #ffffff; border-radius: 12px; padding: 20px; border: 2px solid #F7CAC9; box-shadow: 0 4px 10px rgba(247, 202, 201, 0.12);">
                <h5 style="color: #e11d48; margin-top: 0; margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;">📈 Up-regulated (과발현 - OncomiR)</h5>
                <p style="font-size: 0.88rem; color: #555555; line-height: 1.55; margin-bottom: 0;">
                    암 조직에서 정상 대비 현저히 발현량이 증가한 miRNA입니다. 주로 종양 억제 유전자(Tumor Suppressors, 예: PTEN)를 표적하여 억제함으로써, 세포가 사멸하지 않고 무한 증식하게 도와주는 <b>종양 유발 인자</b> 역할을 합니다. (예: <i>hsa-miR-21, hsa-miR-155</i> 등)
                </p>
            </div>
            <div style="flex: 1; min-width: 300px; background-color: #ffffff; border-radius: 12px; padding: 20px; border: 2px solid #92A8D1; box-shadow: 0 4px 10px rgba(146, 168, 209, 0.12);">
                <h5 style="color: #2563eb; margin-top: 0; margin-bottom: 10px; font-weight: 700; font-size: 1.05rem;">📉 Down-regulated (저발현 - Tumor Suppressor)</h5>
                <p style="font-size: 0.88rem; color: #555555; line-height: 1.55; margin-bottom: 0;">
                    암 조직에서 정상 대비 발현량이 감소한 miRNA입니다. 정상 상태에서는 종양 유발 유전자(Oncogenes, 예: RAS, MYC)를 억제하여 비정상 세포 증식을 막아주는 역할을 하나, 암화 과정에서 소실되어 종양 억제 기능이 약화됩니다. (예: <i>hsa-let-7a</i> 등)
                </p>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # 🔬 유전자 분석 화면 (Analysis)
    # ---------------------------------------------------------
    elif menu == "🔬 유전자 분석 (Analysis)":
        st.markdown('<p class="main-title" style="font-size: 2.2rem;">🔬 유전자 상세 분석</p>', unsafe_allow_html=True)
        
        with st.sidebar:
            st.markdown("### 🎛️ Analysis Controls")
            selected_gene = st.selectbox(
                "타겟 유전자 (Target miRNA)", 
                options=gene_cols_sorted,
                index=0,
                help="분석하고 싶은 miRNA를 검색하거나 선택하세요."
            )
            st.markdown("💡 **Tip**: 우측 상단의 P-value가 **0.05 미만**일 경우 통계적으로 유의미한 발현량 차이가 있다고 해석할 수 있습니다.")

        # 통계 검정 (T-test)
        normal_data = df[df[group_col] == 'Normal'][selected_gene].dropna()
        cancer_data = df[df[group_col] == 'Cancer (Tumor)'][selected_gene].dropna()
        
        t_stat, p_value = stats.ttest_ind(cancer_data, normal_data, equal_var=False)
        
        p_val_display = f"{p_value:.2e}" if p_value < 0.001 else f"{p_value:.4f}"
        p_val_class = "p-value-significant" if p_value < 0.05 else "p-value-nonsignificant"
        sig_text = "유의미한 차이 O" if p_value < 0.05 else "차이 없음 X"

        col1, col2 = st.columns(2)
        with col1:
            avg_diff = cancer_data.mean() - normal_data.mean()
            diff_color = "#F7CAC9" if avg_diff > 0 else "#92A8D1"
            sign = "+" if avg_diff > 0 else ""
            st.markdown(f'''
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value" style="color: {diff_color}; font-size:1.8rem;">{sign}{avg_diff:.2f}</div>
                <div class="metric-label">평균 발현량 차이 (Cancer - Normal)</div>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-card" style="padding:1rem;">
                <div class="metric-value {p_val_class}" style="font-size:1.8rem;">{p_val_display}</div>
                <div class="metric-label">T-test P-value ({sig_text})</div>
            </div>
            ''', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)

        # --- 생명정보학적 해석 (Biological Insights) 섹션 ---
        with st.container(border=True):
            st.markdown(f"#### 🧬 {selected_gene}의 생물학적 특징")
            
            # API 함수를 호출하여 실시간으로 insight 데이터 생성!
            insight = fetch_mirna_insight(selected_gene)
            
            ins_col1, ins_col2 = st.columns([1.5, 1])
            with ins_col1:
                st.markdown(f"""
                <div style='background-color: #f0f4f8; padding: 18px; border-radius: 12px; border-left: 5px solid #92A8D1; margin-bottom: 10px;'>
                    <p style='margin-bottom: 8px;'><b>🧬 핵심 타겟:</b> <span style='color:#e74c3c;'>{insight['target']}</span></p>
                    <p style='margin-bottom: 8px;'><b>🧬 관여 경로 (Pathways):</b></p>
                    <div style='display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 10px;'>
                        {" ".join([f"<span style='background-color:#92A8D1; color:white; padding: 2px 8px; border-radius: 15px; font-size: 0.8rem;'>{p}</span>" for p in insight['pathways']])}
                    </div>
                    <p style='margin-bottom: 8px;'><b>📜 생물학적 기전:</b> {insight['role']}</p>
                    <p style='font-size: 0.85rem; color: #555; line-height: 1.4;'>{insight['desc']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # If the fallback message is shown, display a warning for the user
                if insight['target'] == "타깃 정보를 찾을 수 없습니다. PubMed에서 검색해 보세요.":
                    st.warning("현재 선택된 miRNA에 대한 타깃 정보를 API에서 찾을 수 없습니다. PubMed 검색을 통해 최신 문헌을 확인해 주세요.")
                
                # 질병 이름 매핑을 위한 base_name 선언 (하단 펍메드 링크용)
                base_name = selected_gene.split('(')[0].strip()
                search_url = f"https://pubmed.ncbi.nlm.nih.gov/?term={base_name}+cancer"
                st.markdown(f"🔗 [PubMed에서 {base_name} 관련 최신 논문 검색하기]({search_url})")
            
            with ins_col2:
                # 작은 바이올린 플롯으로 시각적 보조
                fig_mini = px.violin(df, x=group_col, y=selected_gene, color=group_col, box=True, template="plotly_white", color_discrete_map={"Normal": "#92A8D1", "Cancer (Tumor)": "#F7CAC9"})
                fig_mini.update_layout(height=270, margin=dict(l=10, r=10, t=10, b=10), showlegend=False)
                st.plotly_chart(fig_mini, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # --- 볼케이노 플롯을 위한 전전처리 및 계산 ---
        @st.cache_data(show_spinner=False)
        def calculate_all_diff_exp(dataframe):
            """모든 유전자에 대해 Fold Change와 P-value를 미리 계산합니다."""
            g_col = 'target_actual'
            # 분석용 컬럼 제외
            n_gene_cols = ['target', 'target_actual', 'target_original']
            g_cols = [col for col in dataframe.columns if col not in n_gene_cols]
            
            cancer_samples = dataframe[dataframe[g_col] == 'Cancer (Tumor)']
            normal_samples = dataframe[dataframe[g_col] == 'Normal']
            
            res = []
            for gene in g_cols:
                try:
                    c_v = pd.to_numeric(cancer_samples[gene], errors='coerce').dropna()
                    n_v = pd.to_numeric(normal_samples[gene], errors='coerce').dropna()
                    if len(c_v) < 2 or len(n_v) < 2: continue
                    
                    fc = c_v.mean() - n_v.mean()
                    _, p = stats.ttest_ind(c_v, n_v, equal_var=False)
                    if np.isnan(p): continue
                    
                    res.append({'Gene': gene, 'FC': fc, 'P': p, 'logP': -np.log10(p)})
                except: continue
            return pd.DataFrame(res)

        with st.spinner('볼케이노 플롯을 위한 전체 유전자 분석 중... 잠시만요! 🌸'):
            volcano_df = calculate_all_diff_exp(df)

        # --- 유의성 기준 설정 및 상태 계산 (에러 방지를 위해 상단으로 이동) ---
        with st.container(border=True):
            st.markdown("#### ⚙️ 분석 기준 설정")
            v_col1, v_col2, v_col3 = st.columns([1, 1, 2])
            with v_col1:
                p_crit = st.number_input("P-value 기준 (α)", value=0.05, format="%.3f", step=0.005, key="p_crit_val")
            with v_col2:
                fc_crit = st.number_input("Log2 FC 기준", value=1.0, format="%.1f", step=0.1, key="fc_crit_val")
            
            # 유의성 라벨링 미리 수행
            volcano_df['Status'] = 'Non-significant'
            volcano_df.loc[(volcano_df['P'] < p_crit) & (volcano_df['FC'] > fc_crit), 'Status'] = 'Up-regulated'
            volcano_df.loc[(volcano_df['P'] < p_crit) & (volcano_df['FC'] < -fc_crit), 'Status'] = 'Down-regulated'
            
            with v_col3:
                up_count = len(volcano_df[volcano_df['Status'] == 'Up-regulated'])
                down_count = len(volcano_df[volcano_df['Status'] == 'Down-regulated'])
                st.markdown(f"✅ 과발현: **{up_count}개** / 저발현: **{down_count}개** 유전자가 발견되었습니다.")

        col_v1, col_v2 = st.columns([1, 1.2])
        
        # 1. 바이올린 차트 (기존 유지)
        with col_v1:
            with st.container(border=True):
                st.markdown(f"<h4 style='text-align: center; color: #334155;'>📊 발현량 분포: <b>{selected_gene}</b></h4>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 0; margin-bottom: 1rem;'>", unsafe_allow_html=True)
                fig_box = px.violin(
                    df, 
                    x=group_col, 
                    y=selected_gene, 
                    color=group_col,
                    box=True, 
                    points="all", 
                    template="plotly_white",
                    labels={selected_gene: "Expression Level", group_col: "Group"},
                    color_discrete_map={"Normal": "#92A8D1", "Cancer (Tumor)": "#F7CAC9"}
                )
                fig_box.update_layout(
                    margin=dict(l=20, r=20, t=10, b=20), 
                    showlegend=False,
                    xaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)"
                )
                fig_box.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#f1f5f9')
                st.plotly_chart(fig_box, use_container_width=True)

        # 2. 클러스터링 히트맵 구현 (고도화 버전)
        with col_v2:
            with st.container(border=True):
                st.markdown("<h4 style='text-align: center; color: #334155;'>🔥 패턴 클러스터링 히트맵</h4>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 0; margin-bottom: 1rem;'>", unsafe_allow_html=True)
                
                @st.cache_data(show_spinner=False)
                def get_clustered_heatmap_data(dataframe, diff_results, top_n=30):
                    from scipy.cluster.hierarchy import linkage, leaves_list
                    from scipy.spatial.distance import pdist
                    
                    # 1. P-value 기준 가장 유의미한 상위 N개 유전자 선택
                    significant_genes = diff_results.sort_values('P').head(top_n)['Gene'].tolist()
                    
                    # 2. 데이터 추출 및 Z-score 정규화
                    subset = dataframe[significant_genes].apply(pd.to_numeric, errors='coerce').fillna(0)
                    z_data = subset.apply(lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0)
                    
                    # 3. 계층적 클러스터링 수행 (행: 유전자, 열: 샘플)
                    # 유전자 클러스터링 (행 방향)
                    gene_dist = pdist(z_data.T, metric='euclidean')
                    gene_linkage = linkage(gene_dist, method='ward')
                    gene_order = leaves_list(gene_linkage)
                    
                    # 샘플 클러스터링 (열 방향)
                    sample_dist = pdist(z_data, metric='euclidean')
                    sample_linkage = linkage(sample_dist, method='ward')
                    sample_order = leaves_list(sample_linkage)
                    
                    # 4. 정렬된 데이터 생성
                    ordered_genes = [significant_genes[i] for i in gene_order]
                    # 샘플 정렬 및 그룹 정보 유지
                    ordered_z_data = z_data.iloc[sample_order][ordered_genes].T
                    ordered_groups = dataframe.iloc[sample_order]['target_actual'].tolist()
                    
                    return ordered_z_data, ordered_groups, ordered_genes

            try:
                heatmap_z, sample_groups, ordered_genes = get_clustered_heatmap_data(df, volcano_df, top_n=30)
                
                fig_heat = px.imshow(
                    heatmap_z,
                    labels=dict(x="Samples (Clustered)", y="miRNA", color="Z-score"),
                    x=[f"{g}" for g in sample_groups],
                    y=ordered_genes,
                    aspect="auto",
                    color_continuous_scale=[[0, "#92A8D1"], [0.5, "#F0EEEB"], [1, "#F7CAC9"]],
                    color_continuous_midpoint=0
                )
                fig_heat.update_xaxes(showticklabels=True, tickangle=45, tickfont=dict(size=8))
                fig_heat.update_layout(height=500, margin=dict(l=20, r=20, t=20, b=20))
                st.plotly_chart(fig_heat, use_container_width=True)
                
                st.info(f"💡 **분석 결과**: 상위 {len(ordered_genes)}개의 유전자를 클러스터링한 결과입니다.")
                
                # --- 상위 유전자 상세 수치 표 (이제 Status가 확실히 존재합니다) ---
                with st.expander("📊 상위 유전자 상세 수치 보기", expanded=False):
                    top_gene_stats = volcano_df.sort_values('P').head(30)[['Gene', 'FC', 'P', 'Status']]
                    top_gene_stats.columns = ['miRNA 이름', 'Log2 Fold Change', 'P-value', '상태']
                    st.dataframe(top_gene_stats.style.background_gradient(subset=['Log2 Fold Change'], cmap='coolwarm'), use_container_width=True)
            except Exception as e:
                st.error(f"히트맵 생성 중 오류가 발생했어요: {e}")

        # --- 3. Volcano Plot ---
        st.markdown("<br>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<h4 style='text-align: center; color: #334155;'>🌋 유전자 차별 발현 분석 (Volcano Plot)</h4>", unsafe_allow_html=True)
            st.markdown("<hr style='margin-top: 0; margin-bottom: 1rem;'>", unsafe_allow_html=True)
            
            # 아래쪽의 입력창과 상태 계산 로직은 위로 이동했으므로, 여기서는 그래프만 그립니다.
            
            fig_volcano = px.scatter(
                volcano_df, x='FC', y='logP', color='Status',
                hover_name='Gene',
                labels={'FC': 'Log2 Fold Change', 'logP': '-Log10 P-value'},
                color_discrete_map={'Up-regulated': '#F7CAC9', 'Down-regulated': '#92A8D1', 'Non-significant': '#cbd5e1'},
                template="plotly_white"
            )
            # 기준선 추가
            fig_volcano.add_vline(x=fc_crit, line_dash="dash", line_color="#F7CAC9", opacity=0.7)
            fig_volcano.add_vline(x=-fc_crit, line_dash="dash", line_color="#92A8D1", opacity=0.7)
            fig_volcano.add_hline(y=-np.log10(p_crit), line_dash="dash", line_color="#334155", opacity=0.7)
            
            fig_volcano.update_layout(height=550, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_volcano, use_container_width=True)
            
            with v_col3:
                up_count = len(volcano_df[volcano_df['Status'] == 'Up-regulated'])
                down_count = len(volcano_df[volcano_df['Status'] == 'Down-regulated'])
                st.write(f"✅ **결과 요약**: 과발현 유전자 **{up_count}개**, 저발현 유전자 **{down_count}개**가 발견되었습니다.")

        st.markdown("---")
        with st.expander("💾 원본 데이터 확인 및 결과 Export (다운로드)", expanded=False):
            st.markdown("분석된 **Top 10 유전자 발현 패턴(Z-score)**과 **전체 원본 데이터**를 CSV 형식으로 다운로드할 수 있습니다.")
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                # 클러스터링된 히트맵 데이터 CSV 변환
                try:
                    heatmap_csv = heatmap_z.to_csv().encode('utf-8-sig')
                    st.download_button(
                        label="📥 클러스터링 히트맵 데이터 다운로드 (CSV)",
                        data=heatmap_csv,
                        file_name="clustered_heatmap_data.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                except:
                    st.info("히트맵 데이터를 준비 중입니다...")
            
            with dl_col2:
                # 전체 데이터 CSV 변환
                raw_csv = df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 전체 데이터 다운로드 (CSV)",
                    data=raw_csv,
                    file_name="GSE58606_processed.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            st.markdown("<hr style='margin 1rem 0;'>", unsafe_allow_html=True)
            st.markdown("**(미리보기) 데이터 상위 100개 항목**")
            st.dataframe(df[[group_col, selected_gene]].head(100), use_container_width=True)

else:
    st.warning("데이터를 불러오지 못했습니다. 파일 경로(`archive/GSE58606_data.csv`)를 확인해주세요.")
