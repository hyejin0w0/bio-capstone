import streamlit as st
import pandas as pd
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

# 데이터 로딩 함수 (캐싱 적용)
@st.cache_data
def load_data():
    file_path = 'archive/GSE58606_data.csv'
    try:
        df = pd.read_csv(file_path)
        # 중복되는 이름이 있을 경우 (2), (3) 등을 붙여서 고유한 컬럼명 유지
        cleaned_columns = []
        seen = {}
        for col in df.columns:
            if " : " in col:
                clean_name = col.split(" : ")[1].strip()
            else:
                clean_name = col
            
            if clean_name in seen:
                seen[clean_name] += 1
                clean_name = f"{clean_name} ({seen[clean_name]})"
            else:
                seen[clean_name] = 1
                
            cleaned_columns.append(clean_name)
        df.columns = cleaned_columns
        
        # 보기 좋게 target_actual 값을 변경
        if 'target_actual' in df.columns:
            df['target_actual'] = df['target_actual'].replace({
                'primary breast cancer': 'Cancer (Tumor)',
                'normal breast tissue': 'Normal'
            })
            
        return df
    except Exception as e:
        st.error(f"데이터 로딩 오류: {e}")
        return None

df = load_data()

if df is not None:
    # 타겟 컬럼 및 유전자 목록 추출
    group_col = 'target_actual'
    non_gene_cols = ['target', 'target_actual']
    gene_cols = [col for col in df.columns if col not in non_gene_cols]
    gene_cols_sorted = sorted(gene_cols)
    
    normal_count = len(df[df[group_col] == 'Normal'])
    cancer_count = len(df[df[group_col] == 'Cancer (Tumor)'])

    # 사이드바 메뉴 (화면 분리)
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>🧭 Navigation</h2>", unsafe_allow_html=True)
        menu = st.radio("화면 이동", ["🏠 홈 (Overview)", "🔬 유전자 분석 (Analysis)"], label_visibility="collapsed")
        st.markdown("---")

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
        
        col_v1, col_v2 = st.columns([1, 1.2])
        
        # 1. 바이올린 차트 (구획 명확화)
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

        # 2. 히트맵 구현 (구획 명확화 및 TypeError 수정)
        with col_v2:
            with st.container(border=True):
                st.markdown("<h4 style='text-align: center; color: #334155;'>🔥 Top 10 유전자 발현 패턴</h4>", unsafe_allow_html=True)
                st.markdown("<hr style='margin-top: 0; margin-bottom: 1rem;'>", unsafe_allow_html=True)
                
                @st.cache_data
                def get_top_10_heatmap_data(dataframe):
                    normal_means = dataframe[dataframe[group_col] == 'Normal'][gene_cols].mean()
                    cancer_means = dataframe[dataframe[group_col] == 'Cancer (Tumor)'][gene_cols].mean()
                    diffs = cancer_means - normal_means
                    
                    # 절대값 기준으로 가장 차이가 큰 10개 선택
                    top10_genes = diffs.abs().sort_values(ascending=False).head(10).index.tolist()
                    
                    # 히트맵을 위해 데이터 재배열 (정상군이 왼쪽, 환자군이 오른쪽으로 오도록 정렬)
                    sorted_df = dataframe.sort_values(by=group_col, ascending=False)
                    heatmap_data = sorted_df[top10_genes].T # 유전자가 행, 샘플이 열
                    
                    # Z-score 정규화 (유전자별로 시각적 대비를 극대화하기 위함)
                    # Pandas Series of arrays 대신 2D DataFrame이 유지되도록 수정 (TypeError 해결)
                    heatmap_data = pd.DataFrame(
                        stats.zscore(heatmap_data, axis=1), 
                        index=heatmap_data.index, 
                        columns=heatmap_data.columns
                    )
                    return heatmap_data, sorted_df[group_col].tolist()

            heatmap_data, sample_groups = get_top_10_heatmap_data(df)
            
            fig_heat = px.imshow(
                heatmap_data,
                labels=dict(x="Patients/Samples", y="miRNA", color="Expression (Z-score)"),
                x=[f"Sample_{i}" for i in range(len(sample_groups))],
                y=heatmap_data.index,
                aspect="auto",
                color_continuous_scale=[[0, "#92A8D1"], [0.5, "#F0EEEB"], [1, "#F7CAC9"]],
                color_continuous_midpoint=0
            )
            
            # X축 레이블을 없애고 레이아웃 다듬기
            fig_heat.update_xaxes(showticklabels=False)
            fig_heat.update_layout(margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # 해석 가이드 (Alert Box 형태)
            st.info('''
            **[히트맵 해석 가이드]**
            * **가로축(X축)**은 133개의 전체 샘플을 나타내며, 왼쪽에는 '정상(Normal)', 오른쪽에는 '암(Cancer)' 샘플들이 모여 있습니다.
            * **세로축(Y축)**은 암과 정상 조직 간에 가장 차이가 심한 상위 10개의 바이오마커 유전자입니다.
            * **색상(Color)**: 붉은색일수록 해당 샘플에서 그 유전자가 평균보다 많이 발현되었음(과발현)을 의미하고, 푸른색일수록 적게 발현되었음(저발현)을 의미합니다.
            * **결과 요약**: 붉은색과 푸른색 영역이 좌우로 뚜렷하게 나뉘어 보인다면, 해당 유전자들이 정상 조직과 암 조직을 완벽하게 구분해내는 핵심 유전자임을 증명합니다.
            ''')
            
        st.markdown("---")
        with st.expander("💾 원본 데이터 확인 및 결과 Export (다운로드)", expanded=False):
            st.markdown("분석된 **Top 10 유전자 발현 패턴(Z-score)**과 **전체 원본 데이터**를 CSV 형식으로 다운로드할 수 있습니다. 다운로드된 파일은 엑셀(Excel)에서 열어서 다른 사람과 공유하거나 논문/발표 자료용으로 활용할 수 있습니다.")
            
            # 다운로드 버튼 가로 배치
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                # Top 10 히트맵 데이터 CSV 변환
                heatmap_csv = heatmap_data.to_csv().encode('utf-8-sig') # Excel에서 한글/특수문자 깨짐 방지
                st.download_button(
                    label="📥 Top 10 유전자 발현 패턴 다운로드 (CSV)",
                    data=heatmap_csv,
                    file_name="top10_heatmap_expression.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
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
                
            st.markdown("<hr style='margin: 1rem 0;'>", unsafe_allow_html=True)
            st.markdown("**(미리보기) 데이터 상위 100개 항목**")
            st.dataframe(df[[group_col, selected_gene]].head(100), use_container_width=True)

else:
    st.warning("데이터를 불러오지 못했습니다. 파일 경로(`archive/GSE58606_data.csv`)를 확인해주세요.")
