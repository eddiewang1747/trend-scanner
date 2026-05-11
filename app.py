"""
US Stock Trend Scanner - 趋势感知模型
三层漏斗: 宏观资金 -> 板块强度 -> 个股形态
作为 2560 战法系统的上游过滤器
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ============ 页面配置 ============
st.set_page_config(
    page_title="Trend Scanner | 趋势扫描",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============ 自定义样式 (Bloomberg Terminal 风格) ============
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&family=Inter:wght@400;600;800&display=swap');
    
    .stApp {
        background-color: #0a0e1a;
        color: #e8e8e8;
    }
    
    .main-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 28px;
        font-weight: 700;
        color: #ff9500;
        letter-spacing: 2px;
        border-bottom: 2px solid #ff9500;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 14px;
        font-weight: 700;
        color: #00d4ff;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: 20px;
        margin-bottom: 8px;
        padding: 4px 8px;
        background: linear-gradient(90deg, #00d4ff22 0%, transparent 100%);
        border-left: 3px solid #00d4ff;
    }
    
    .metric-box {
        background-color: #111729;
        border: 1px solid #1f2937;
        border-radius: 4px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .signal-strong { color: #00ff88; font-weight: 700; }
    .signal-weak { color: #ff3366; font-weight: 700; }
    .signal-neutral { color: #ffaa00; font-weight: 700; }
    
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        background-color: #111729;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #888;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ff9500 !important;
        background-color: #1a2138 !important;
    }
    
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 20px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-family: 'Inter', sans-serif;
        font-size: 11px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #888;
    }
</style>
""", unsafe_allow_html=True)

# ============ 数据定义 ============

# 11 个 SPDR 行业 ETF + 主题 ETF
SECTOR_ETFS = {
    'XLK': 'Technology 科技',
    'XLF': 'Financials 金融',
    'XLV': 'Healthcare 医疗',
    'XLE': 'Energy 能源',
    'XLI': 'Industrials 工业',
    'XLY': 'Cons. Discretionary 非必需消费',
    'XLP': 'Cons. Staples 必需消费',
    'XLU': 'Utilities 公用',
    'XLB': 'Materials 材料',
    'XLRE': 'Real Estate 地产',
    'XLC': 'Communication 通信',
}

THEME_ETFS = {
    'SMH': 'Semiconductors 半导体',
    'XBI': 'Biotech 生物科技',
    'ITA': 'Aerospace/Defense 军工',
    'KWEB': 'China Internet 中概互联',
    'URA': 'Uranium 铀矿',
    'ICLN': 'Clean Energy 清洁能源',
    'ARKK': 'Innovation 创新',
    'JETS': 'Airlines 航空',
    'XME': 'Metals/Mining 金属矿业',
    'IBB': 'Biotech (Large) 大生物',
    'SOXX': 'Semis (Alt) 半导体II',
    'TAN': 'Solar 太阳能',
    'LIT': 'Lithium 锂电池',
    'BOTZ': 'Robotics 机器人',
    'HACK': 'Cybersecurity 网络安全',
    'FINX': 'Fintech 金融科技',
    'BLOK': 'Blockchain 区块链',
    'GDX': 'Gold Miners 金矿',
}

ALL_ETFS = {**SECTOR_ETFS, **THEME_ETFS}

# 各板块龙头股 (用于第二层验证)
SECTOR_LEADERS = {
    'XLK': ['AAPL', 'MSFT', 'NVDA', 'AVGO', 'ORCL'],
    'XLF': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
    'XLV': ['LLY', 'UNH', 'JNJ', 'ABBV', 'MRK'],
    'XLE': ['XOM', 'CVX', 'COP', 'EOG', 'SLB'],
    'XLI': ['GE', 'CAT', 'RTX', 'HON', 'UNP'],
    'XLY': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
    'XLP': ['WMT', 'PG', 'COST', 'KO', 'PEP'],
    'XLU': ['NEE', 'SO', 'DUK', 'CEG', 'AEP'],
    'XLB': ['LIN', 'SHW', 'APD', 'ECL', 'FCX'],
    'XLRE': ['PLD', 'AMT', 'EQIX', 'WELL', 'CCI'],
    'XLC': ['META', 'GOOGL', 'NFLX', 'DIS', 'TMUS'],
    'SMH': ['NVDA', 'TSM', 'AVGO', 'AMD', 'ASML'],
    'XBI': ['VRTX', 'REGN', 'GILD', 'MRNA', 'BIIB'],
    'ITA': ['BA', 'RTX', 'LMT', 'GD', 'NOC'],
    'KWEB': ['BABA', 'PDD', 'JD', 'BIDU', 'NTES'],
    'URA': ['CCJ', 'UEC', 'DNN', 'NXE', 'UUUU'],
    'ICLN': ['ENPH', 'FSLR', 'PLUG', 'SEDG', 'RUN'],
    'ARKK': ['TSLA', 'COIN', 'ROKU', 'PATH', 'HOOD'],
    'JETS': ['DAL', 'UAL', 'AAL', 'LUV', 'ALK'],
    'XME': ['FCX', 'NEM', 'NUE', 'STLD', 'AA'],
    'IBB': ['VRTX', 'REGN', 'AMGN', 'GILD', 'MRNA'],
    'SOXX': ['NVDA', 'AVGO', 'AMD', 'QCOM', 'INTC'],
    'TAN': ['FSLR', 'ENPH', 'SEDG', 'RUN', 'NXT'],
    'LIT': ['ALB', 'SQM', 'TSLA', 'PCRFY', 'LAC'],
    'BOTZ': ['NVDA', 'ABB', 'ISRG', 'KEYS', 'IRBT'],
    'HACK': ['CRWD', 'PANW', 'FTNT', 'ZS', 'OKTA'],
    'FINX': ['V', 'MA', 'PYPL', 'COIN', 'HOOD'],
    'BLOK': ['COIN', 'MSTR', 'MARA', 'RIOT', 'HUT'],
    'GDX': ['NEM', 'GOLD', 'AEM', 'WPM', 'FNV'],
}

# ============ 数据获取函数 ============

@st.cache_data(ttl=3600)
def fetch_price_data(tickers, period='6mo'):
    """批量获取价格数据"""
    try:
        data = yf.download(
            tickers=tickers if isinstance(tickers, list) else [tickers],
            period=period,
            interval='1d',
            group_by='ticker',
            auto_adjust=True,
            progress=False,
            threads=True
        )
        return data
    except Exception as e:
        st.error(f"数据获取失败: {e}")
        return None

@st.cache_data(ttl=3600)
def get_etf_metrics(ticker):
    """获取单个 ETF 的关键指标"""
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period='1y')
        if hist.empty:
            return None
        
        close = hist['Close']
        volume = hist['Volume']
        
        # 收益率
        ret_1w = (close.iloc[-1] / close.iloc[-6] - 1) * 100 if len(close) >= 6 else 0
        ret_1m = (close.iloc[-1] / close.iloc[-22] - 1) * 100 if len(close) >= 22 else 0
        ret_3m = (close.iloc[-1] / close.iloc[-66] - 1) * 100 if len(close) >= 66 else 0
        ret_6m = (close.iloc[-1] / close.iloc[-132] - 1) * 100 if len(close) >= 132 else 0
        
        # 距离52周高点
        high_52w = close.tail(252).max() if len(close) >= 252 else close.max()
        pct_from_high = (close.iloc[-1] / high_52w - 1) * 100
        
        # 成交量趋势 (近5日均量 / 60日均量)
        vol_5 = volume.tail(5).mean()
        vol_60 = volume.tail(60).mean()
        vol_ratio = vol_5 / vol_60 if vol_60 > 0 else 1
        
        # MA 状态
        ma25 = close.rolling(25).mean().iloc[-1]
        ma50 = close.rolling(50).mean().iloc[-1]
        ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
        
        above_ma25 = close.iloc[-1] > ma25
        above_ma50 = close.iloc[-1] > ma50
        above_ma200 = close.iloc[-1] > ma200
        ma_aligned = ma25 > ma50 > ma200  # 多头排列
        
        return {
            'price': close.iloc[-1],
            'ret_1w': ret_1w,
            'ret_1m': ret_1m,
            'ret_3m': ret_3m,
            'ret_6m': ret_6m,
            'pct_from_52w_high': pct_from_high,
            'vol_ratio': vol_ratio,
            'above_ma25': above_ma25,
            'above_ma50': above_ma50,
            'above_ma200': above_ma200,
            'ma_aligned': ma_aligned,
        }
    except Exception as e:
        return None

@st.cache_data(ttl=3600)
def compute_relative_strength(ticker, benchmark='SPY', period='3mo'):
    """计算相对 SPY 的强度"""
    try:
        data = yf.download([ticker, benchmark], period=period, progress=False, auto_adjust=True)['Close']
        if data.empty or len(data) < 20:
            return 0
        
        ticker_ret = (data[ticker].iloc[-1] / data[ticker].iloc[0] - 1) * 100
        bench_ret = (data[benchmark].iloc[-1] / data[benchmark].iloc[0] - 1) * 100
        return ticker_ret - bench_ret
    except:
        return 0

def calculate_trend_score(metrics, rs):
    """综合趋势打分 (0-100)"""
    if metrics is None:
        return 0
    
    score = 0
    # 动量得分 (40分)
    if metrics['ret_1m'] > 0: score += 8
    if metrics['ret_3m'] > 0: score += 12
    if metrics['ret_1w'] > 0: score += 5
    if metrics['ret_1m'] > 5: score += 5
    if metrics['ret_3m'] > 10: score += 10
    
    # 趋势结构 (30分)
    if metrics['above_ma25']: score += 8
    if metrics['above_ma50']: score += 10
    if metrics['above_ma200']: score += 5
    if metrics['ma_aligned']: score += 7
    
    # 成交量 (15分)
    if metrics['vol_ratio'] > 1.0: score += 5
    if metrics['vol_ratio'] > 1.3: score += 5
    if metrics['vol_ratio'] > 1.6: score += 5
    
    # 相对强度 (15分)
    if rs > 0: score += 5
    if rs > 5: score += 5
    if rs > 10: score += 5
    
    return min(score, 100)

def signal_color(score):
    """根据分数返回颜色"""
    if score >= 75: return '🟢', 'signal-strong'
    if score >= 50: return '🟡', 'signal-neutral'
    return '🔴', 'signal-weak'

# ============ UI 主体 ============

st.markdown('<div class="main-header">▲ TREND SCANNER · 趋势感知模型</div>', unsafe_allow_html=True)
st.markdown(f"<div style='font-family: JetBrains Mono; color: #888; font-size: 12px; margin-bottom: 20px;'>三层漏斗: 宏观资金 → 板块强度 → 个股形态  |  数据更新: {datetime.now().strftime('%Y-%m-%d %H:%M')} ET</div>", unsafe_allow_html=True)

# ============ Sidebar 控制 ============
with st.sidebar:
    st.markdown("### ⚙ 扫描配置")
    
    scan_mode = st.radio(
        "扫描模式",
        ["完整扫描 (慢)", "仅核心行业 ETF", "仅主题 ETF"],
        help="完整扫描覆盖所有 ETF，约需 30-60 秒"
    )
    
    min_score = st.slider(
        "最低趋势分数",
        0, 100, 60,
        help="只显示分数高于此值的板块"
    )
    
    st.markdown("---")
    st.markdown("### 📊 模型参数")
    
    rs_period = st.selectbox(
        "相对强度周期",
        ['1mo', '3mo', '6mo'],
        index=1
    )
    
    st.markdown("---")
    st.markdown("""
    ### 📖 模型说明
    
    **第一层** 板块趋势打分  
    动量(40) + 结构(30) + 量能(15) + RS(15)
    
    **第二层** 龙头股验证  
    板块龙头 RS 一致性
    
    **第三层** 个股 2560 信号  
    对接你的扫描器输出
    
    ---
    
    **信号强度**  
    🟢 ≥75 强势趋势  
    🟡 50-74 形成中  
    🔴 <50 弱势
    """)
    
    refresh = st.button("🔄 重新扫描", use_container_width=True)

if refresh:
    st.cache_data.clear()

# ============ 决定扫描范围 ============
if scan_mode == "完整扫描 (慢)":
    scan_etfs = ALL_ETFS
elif scan_mode == "仅核心行业 ETF":
    scan_etfs = SECTOR_ETFS
else:
    scan_etfs = THEME_ETFS

# ============ Tabs ============
tab1, tab2, tab3, tab4 = st.tabs([
    "🌐 第一层 · 板块资金趋势",
    "🎯 第二层 · 龙头股验证",
    "📈 第三层 · 个股观察名单",
    "🗺 相对强度地图"
])

# ============ TAB 1: 第一层扫描 ============
with tab1:
    st.markdown('<div class="section-header">▌ 板块趋势打分排名</div>', unsafe_allow_html=True)
    
    with st.spinner("⟳ 扫描中... 拉取 ETF 数据..."):
        results = []
        progress = st.progress(0)
        total = len(scan_etfs)
        
        for i, (ticker, name) in enumerate(scan_etfs.items()):
            metrics = get_etf_metrics(ticker)
            if metrics is None:
                continue
            rs = compute_relative_strength(ticker, 'SPY', rs_period)
            score = calculate_trend_score(metrics, rs)
            
            results.append({
                'Ticker': ticker,
                'Sector': name,
                'Score': score,
                'Price': metrics['price'],
                '1W%': metrics['ret_1w'],
                '1M%': metrics['ret_1m'],
                '3M%': metrics['ret_3m'],
                '6M%': metrics['ret_6m'],
                'vs SPY': rs,
                'From 52W High': metrics['pct_from_52w_high'],
                'Vol Ratio': metrics['vol_ratio'],
                'MA Aligned': '✓' if metrics['ma_aligned'] else '✗',
                '>MA50': '✓' if metrics['above_ma50'] else '✗',
                '>MA200': '✓' if metrics['above_ma200'] else '✗',
            })
            progress.progress((i + 1) / total)
        
        progress.empty()
    
    if results:
        df = pd.DataFrame(results).sort_values('Score', ascending=False)
        df_filtered = df[df['Score'] >= min_score].copy()
        
        # 顶部指标
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            strong = len(df[df['Score'] >= 75])
            st.metric("强势板块", f"{strong}", f"≥75 分")
        with col2:
            forming = len(df[(df['Score'] >= 50) & (df['Score'] < 75)])
            st.metric("形成中", f"{forming}", "50-74 分")
        with col3:
            weak = len(df[df['Score'] < 50])
            st.metric("弱势板块", f"{weak}", "<50 分")
        with col4:
            avg = df['Score'].mean()
            st.metric("市场广度", f"{avg:.1f}", "平均分")
        
        st.markdown("---")
        
        # 主表格
        def color_score(val):
            if val >= 75: return 'background-color: #00ff8822; color: #00ff88; font-weight: 700'
            if val >= 50: return 'background-color: #ffaa0022; color: #ffaa00; font-weight: 700'
            return 'background-color: #ff336622; color: #ff3366'
        
        def color_pct(val):
            if val > 0: return 'color: #00ff88'
            if val < 0: return 'color: #ff3366'
            return 'color: #888'
        
        styled = df_filtered.style.format({
            'Score': '{:.0f}',
            'Price': '${:.2f}',
            '1W%': '{:+.2f}%',
            '1M%': '{:+.2f}%',
            '3M%': '{:+.2f}%',
            '6M%': '{:+.2f}%',
            'vs SPY': '{:+.2f}',
            'From 52W High': '{:.2f}%',
            'Vol Ratio': '{:.2f}x',
        }).map(color_score, subset=['Score']) \
          .map(color_pct, subset=['1W%', '1M%', '3M%', '6M%', 'vs SPY'])
        
        st.dataframe(styled, use_container_width=True, height=600, hide_index=True)
        
        # 保存结果到 session
        st.session_state['layer1_results'] = df_filtered
        
        # 顶部 3 个板块的可视化
        st.markdown('<div class="section-header">▌ TOP 5 板块走势对比 (相对 SPY)</div>', unsafe_allow_html=True)
        top5 = df_filtered.head(5)['Ticker'].tolist()
        if top5:
            comparison_data = yf.download(top5 + ['SPY'], period='6mo', progress=False, auto_adjust=True)['Close']
            # 归一化到 100
            normalized = comparison_data / comparison_data.iloc[0] * 100
            
            fig = go.Figure()
            colors = ['#ff9500', '#00d4ff', '#00ff88', '#ff3366', '#aa88ff', '#888888']
            for i, col in enumerate(normalized.columns):
                fig.add_trace(go.Scatter(
                    x=normalized.index,
                    y=normalized[col],
                    name=col,
                    line=dict(color=colors[i % len(colors)], width=2 if col != 'SPY' else 1.5, dash='dash' if col == 'SPY' else 'solid')
                ))
            
            fig.update_layout(
                template='plotly_dark',
                plot_bgcolor='#0a0e1a',
                paper_bgcolor='#0a0e1a',
                font=dict(family='JetBrains Mono', color='#e8e8e8'),
                hovermode='x unified',
                height=400,
                margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(gridcolor='#1f2937'),
                yaxis=dict(gridcolor='#1f2937', title='归一化值 (起点=100)')
            )
            st.plotly_chart(fig, use_container_width=True)

# ============ TAB 2: 第二层 - 龙头股验证 ============
with tab2:
    st.markdown('<div class="section-header">▌ 板块龙头股一致性验证</div>', unsafe_allow_html=True)
    st.markdown("<div style='color: #888; font-size: 12px; margin-bottom: 16px;'>检验候选板块内部龙头是否同步走强 — 60% 以上龙头同步 = 趋势确认</div>", unsafe_allow_html=True)
    
    if 'layer1_results' not in st.session_state or st.session_state['layer1_results'].empty:
        st.warning("⚠ 请先在第一层扫描后再查看龙头股验证")
    else:
        # 选择板块
        candidate_sectors = st.session_state['layer1_results']['Ticker'].tolist()[:10]
        
        if not candidate_sectors:
            st.info("没有符合分数阈值的板块，请降低 sidebar 的最低分数")
        else:
            selected_sector = st.selectbox(
                "选择候选板块",
                candidate_sectors,
                format_func=lambda x: f"{x} - {ALL_ETFS.get(x, x)}"
            )
            
            leaders = SECTOR_LEADERS.get(selected_sector, [])
            
            if not leaders:
                st.info("该 ETF 暂无龙头股映射")
            else:
                with st.spinner(f"⟳ 检验 {selected_sector} 板块龙头..."):
                    leader_results = []
                    for stock in leaders:
                        metrics = get_etf_metrics(stock)
                        if metrics is None:
                            continue
                        rs = compute_relative_strength(stock, 'SPY', rs_period)
                        score = calculate_trend_score(metrics, rs)
                        
                        leader_results.append({
                            'Stock': stock,
                            'Score': score,
                            'Price': metrics['price'],
                            '1M%': metrics['ret_1m'],
                            '3M%': metrics['ret_3m'],
                            'vs SPY': rs,
                            '>MA50': '✓' if metrics['above_ma50'] else '✗',
                            'MA Aligned': '✓' if metrics['ma_aligned'] else '✗',
                            'Vol Ratio': metrics['vol_ratio'],
                            'From High': metrics['pct_from_52w_high'],
                        })
                
                if leader_results:
                    leader_df = pd.DataFrame(leader_results).sort_values('Score', ascending=False)
                    
                    # 一致性分析
                    strong_count = len(leader_df[leader_df['Score'] >= 60])
                    total_count = len(leader_df)
                    consistency = strong_count / total_count * 100
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("龙头总数", total_count)
                    with col2:
                        st.metric("强势龙头", f"{strong_count}/{total_count}")
                    with col3:
                        sig = "✅ 趋势确认" if consistency >= 60 else "⚠ 内部分化"
                        st.metric("一致性", f"{consistency:.0f}%", sig)
                    
                    st.markdown("---")
                    
                    styled_leaders = leader_df.style.format({
                        'Score': '{:.0f}',
                        'Price': '${:.2f}',
                        '1M%': '{:+.2f}%',
                        '3M%': '{:+.2f}%',
                        'vs SPY': '{:+.2f}',
                        'Vol Ratio': '{:.2f}x',
                        'From High': '{:.2f}%',
                    }).map(
                        lambda v: 'background-color: #00ff8822; color: #00ff88; font-weight: 700' if v >= 75
                        else ('background-color: #ffaa0022; color: #ffaa00' if v >= 50 else 'background-color: #ff336622; color: #ff3366'),
                        subset=['Score']
                    )
                    
                    st.dataframe(styled_leaders, use_container_width=True, hide_index=True)
                    
                    # 走势对比
                    st.markdown('<div class="section-header">▌ 龙头股走势图 (3M)</div>', unsafe_allow_html=True)
                    stocks_data = yf.download(leaders + [selected_sector], period='3mo', progress=False, auto_adjust=True)['Close']
                    normalized = stocks_data / stocks_data.iloc[0] * 100
                    
                    fig = go.Figure()
                    for col in normalized.columns:
                        is_etf = col == selected_sector
                        fig.add_trace(go.Scatter(
                            x=normalized.index,
                            y=normalized[col],
                            name=col,
                            line=dict(
                                width=3 if is_etf else 1.5,
                                color='#ff9500' if is_etf else None,
                                dash='solid' if is_etf else 'solid'
                            )
                        ))
                    
                    fig.update_layout(
                        template='plotly_dark',
                        plot_bgcolor='#0a0e1a',
                        paper_bgcolor='#0a0e1a',
                        font=dict(family='JetBrains Mono', color='#e8e8e8'),
                        height=400,
                        hovermode='x unified',
                        margin=dict(l=0, r=0, t=20, b=0),
                        xaxis=dict(gridcolor='#1f2937'),
                        yaxis=dict(gridcolor='#1f2937', title='归一化 (起点=100)')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 保存到 session
                    st.session_state['layer2_sector'] = selected_sector
                    st.session_state['layer2_leaders'] = leader_df

# ============ TAB 3: 第三层 - 个股观察名单 ============
with tab3:
    st.markdown('<div class="section-header">▌ 个股观察名单 (对接 2560 战法)</div>', unsafe_allow_html=True)
    st.markdown("<div style='color: #888; font-size: 12px; margin-bottom: 16px;'>从强势板块龙头中筛选符合 Minervini Stage 2 / Qullamaggie EP 形态的标的</div>", unsafe_allow_html=True)
    
    if 'layer1_results' not in st.session_state:
        st.warning("⚠ 请先完成第一层扫描")
    else:
        # 自动汇总所有强势板块的龙头
        strong_sectors = st.session_state['layer1_results'][st.session_state['layer1_results']['Score'] >= 70]['Ticker'].tolist()
        
        if not strong_sectors:
            st.info("当前没有 ≥70 分的强势板块，建议降低分数阈值或等待趋势成形")
        else:
            st.markdown(f"**来自 {len(strong_sectors)} 个强势板块的候选股票池**")
            
            all_candidates = set()
            for sec in strong_sectors:
                all_candidates.update(SECTOR_LEADERS.get(sec, []))
            
            candidates = list(all_candidates)
            
            with st.spinner(f"⟳ 评估 {len(candidates)} 只候选股票..."):
                stock_results = []
                for stock in candidates:
                    metrics = get_etf_metrics(stock)
                    if metrics is None:
                        continue
                    rs = compute_relative_strength(stock, 'SPY', rs_period)
                    score = calculate_trend_score(metrics, rs)
                    
                    # Minervini Stage 2 检查
                    stage2 = (
                        metrics['above_ma25'] and 
                        metrics['above_ma50'] and 
                        metrics['above_ma200'] and
                        metrics['ma_aligned'] and
                        metrics['pct_from_52w_high'] > -25
                    )
                    
                    stock_results.append({
                        'Stock': stock,
                        'Score': score,
                        'Stage 2': '✓' if stage2 else '✗',
                        'Price': metrics['price'],
                        '1W%': metrics['ret_1w'],
                        '1M%': metrics['ret_1m'],
                        '3M%': metrics['ret_3m'],
                        'vs SPY': rs,
                        'From 52W': metrics['pct_from_52w_high'],
                        'Vol 5/60': metrics['vol_ratio'],
                        '>MA50': '✓' if metrics['above_ma50'] else '✗',
                        'MA Aligned': '✓' if metrics['ma_aligned'] else '✗',
                    })
                
                if stock_results:
                    stocks_df = pd.DataFrame(stock_results).sort_values('Score', ascending=False)
                    
                    # 入场清单 (高分 + Stage 2)
                    watchlist = stocks_df[(stocks_df['Score'] >= 70) & (stocks_df['Stage 2'] == '✓')]
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("候选总数", len(stocks_df))
                    with col2:
                        st.metric("Stage 2 标的", len(stocks_df[stocks_df['Stage 2'] == '✓']))
                    with col3:
                        st.metric("最终观察名单", len(watchlist), "≥70分 + Stage2")
                    
                    if not watchlist.empty:
                        st.markdown('<div class="section-header">▌ ⭐ 最终观察名单</div>', unsafe_allow_html=True)
                        styled_wl = watchlist.style.format({
                            'Score': '{:.0f}',
                            'Price': '${:.2f}',
                            '1W%': '{:+.2f}%',
                            '1M%': '{:+.2f}%',
                            '3M%': '{:+.2f}%',
                            'vs SPY': '{:+.2f}',
                            'From 52W': '{:.2f}%',
                            'Vol 5/60': '{:.2f}x',
                        }).map(
                            lambda v: 'background-color: #00ff8822; color: #00ff88; font-weight: 700',
                            subset=['Score']
                        )
                        st.dataframe(styled_wl, use_container_width=True, hide_index=True)
                        
                        # 导出
                        csv = watchlist.to_csv(index=False)
                        st.download_button(
                            "📥 导出观察名单 CSV (喂给 2560 扫描器)",
                            csv,
                            f"watchlist_{datetime.now().strftime('%Y%m%d')}.csv",
                            "text/csv",
                            use_container_width=True
                        )
                    
                    st.markdown('<div class="section-header">▌ 完整候选列表</div>', unsafe_allow_html=True)
                    styled_all = stocks_df.style.format({
                        'Score': '{:.0f}',
                        'Price': '${:.2f}',
                        '1W%': '{:+.2f}%',
                        '1M%': '{:+.2f}%',
                        '3M%': '{:+.2f}%',
                        'vs SPY': '{:+.2f}',
                        'From 52W': '{:.2f}%',
                        'Vol 5/60': '{:.2f}x',
                    }).map(
                        lambda v: 'background-color: #00ff8822; color: #00ff88; font-weight: 700' if v >= 75
                        else ('background-color: #ffaa0022; color: #ffaa00' if v >= 50 else ''),
                        subset=['Score']
                    )
                    st.dataframe(styled_all, use_container_width=True, hide_index=True, height=500)

# ============ TAB 4: 相对强度热力图 ============
with tab4:
    st.markdown('<div class="section-header">▌ 全市场板块相对强度热力图</div>', unsafe_allow_html=True)
    st.markdown("<div style='color: #888; font-size: 12px; margin-bottom: 16px;'>横轴: 3个月动量  |  纵轴: 1个月动量  |  气泡大小: 成交量比  |  颜色: 趋势分数</div>", unsafe_allow_html=True)
    
    if 'layer1_results' in st.session_state:
        df = st.session_state['layer1_results']
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df['3M%'],
            y=df['1M%'],
            mode='markers+text',
            text=df['Ticker'],
            textposition='top center',
            textfont=dict(family='JetBrains Mono', size=11, color='#e8e8e8'),
            marker=dict(
                size=df['Vol Ratio'] * 20,
                color=df['Score'],
                colorscale=[[0, '#ff3366'], [0.5, '#ffaa00'], [1, '#00ff88']],
                showscale=True,
                colorbar=dict(title='Score', titlefont=dict(family='JetBrains Mono')),
                line=dict(color='#0a0e1a', width=1),
                opacity=0.85
            ),
            customdata=df[['Sector', 'Score', 'vs SPY']],
            hovertemplate='<b>%{text}</b><br>' +
                          '%{customdata[0]}<br>' +
                          'Score: %{customdata[1]:.0f}<br>' +
                          '1M: %{y:.2f}%<br>' +
                          '3M: %{x:.2f}%<br>' +
                          'vs SPY: %{customdata[2]:.2f}<extra></extra>'
        ))
        
        # 添加象限线
        fig.add_hline(y=0, line=dict(color='#444', width=1, dash='dash'))
        fig.add_vline(x=0, line=dict(color='#444', width=1, dash='dash'))
        
        # 象限标注
        fig.add_annotation(x=df['3M%'].max() * 0.7, y=df['1M%'].max() * 0.9,
                           text='🔥 强趋势', showarrow=False,
                           font=dict(family='JetBrains Mono', color='#00ff88', size=14))
        fig.add_annotation(x=df['3M%'].min() * 0.7, y=df['1M%'].max() * 0.9,
                           text='↗ 拐点信号', showarrow=False,
                           font=dict(family='JetBrains Mono', color='#00d4ff', size=14))
        fig.add_annotation(x=df['3M%'].max() * 0.7, y=df['1M%'].min() * 0.9,
                           text='↘ 高位回调', showarrow=False,
                           font=dict(family='JetBrains Mono', color='#ffaa00', size=14))
        fig.add_annotation(x=df['3M%'].min() * 0.7, y=df['1M%'].min() * 0.9,
                           text='❄ 弱势', showarrow=False,
                           font=dict(family='JetBrains Mono', color='#ff3366', size=14))
        
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#0a0e1a',
            paper_bgcolor='#0a0e1a',
            font=dict(family='JetBrains Mono', color='#e8e8e8'),
            height=600,
            xaxis=dict(title='3 Month Return (%)', gridcolor='#1f2937', zerolinecolor='#444'),
            yaxis=dict(title='1 Month Return (%)', gridcolor='#1f2937', zerolinecolor='#444'),
            margin=dict(l=0, r=0, t=20, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        <div style='background-color: #111729; padding: 16px; border-left: 3px solid #00d4ff; margin-top: 16px; font-family: JetBrains Mono; font-size: 12px;'>
        <b style='color: #00d4ff;'>象限解读</b><br>
        <b style='color: #00ff88;'>右上 - 强趋势</b>: 1M 和 3M 双正，机构正在配置，可顺势<br>
        <b style='color: #00d4ff;'>左上 - 拐点信号</b>: 3M 负但 1M 转正，趋势刚启动，<i>最有 alpha 但风险高</i><br>
        <b style='color: #ffaa00;'>右下 - 高位回调</b>: 3M 强但 1M 转弱，需观察是否结构破坏<br>
        <b style='color: #ff3366;'>左下 - 弱势</b>: 双负，远离
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("请先运行第一层扫描")

# ============ 底部说明 ============
st.markdown("---")
st.markdown("""
<div style='font-family: JetBrains Mono; color: #555; font-size: 11px; text-align: center; padding: 16px;'>
TREND SCANNER v1.0  ·  数据源: Yahoo Finance  ·  缓存 1 小时  ·  仅供研究参考，不构成投资建议
</div>
""", unsafe_allow_html=True)
