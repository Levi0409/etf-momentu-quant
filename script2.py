import streamlit as st
import pandas as pd
import numpy as np
import akshare as ak
from datetime import datetime, timedelta

# 设置页面宽屏
st.set_page_config(page_title="ETF动量轮动量化系统", layout="wide")

st.title("📈 ETF动量轮动策略回测与信号系统")
st.markdown("**策略逻辑**：每周五计算过去20个交易日动量，等权持有排名前2的ETF。若所有ETF动量为负，则空仓。")

# ================= 1. 参数设置 =================
st.sidebar.header("⚙️ 策略参数")
etf_pool = {
    '510300': '沪深300ETF',
    '510500': '中证500ETF',
    '159915': '创业板ETF',
    '588000': '科创50ETF',
    '510880': '红利ETF',
    '512880': '证券ETF'
}

lookback_days = st.sidebar.slider("动量回看天数 (交易日)", min_value=10, max_value=60, value=20)
top_n = st.sidebar.slider("持有排名前N只", min_value=1, max_value=4, value=2)
start_date = st.sidebar.date_input("回测开始日期", datetime(2022, 1, 1))
end_date = st.sidebar.date_input("回测结束日期", datetime.today())

# ================= 2. 数据获取与处理 =================
@st.cache_data(ttl=3600) # 缓存数据，避免每次刷新都重新请求
def fetch_data(start, end):
    data_dict = {}
    for code, name in etf_pool.items():
        try:
            # akshare获取ETF日线数据
            df = ak.fund_etf_hist_em(symbol=code, period="daily", start_date=start.strftime("%Y%m%d"), end_date=end.strftime("%Y%m%d"), adjust="qfq")
            df['日期'] = pd.to_datetime(df['日期'])
            df.set_index('日期', inplace=True)
            data_dict[code] = df['收盘']
        except Exception as e:
            st.warning(f"获取 {name} 数据失败: {e}")
    return data_dict

with st.spinner("正在拉取ETF行情数据，请稍候..."):
    price_dict = fetch_data(start_date, end_date)

if not price_dict:
    st.error("数据加载失败，请检查网络或AkShare接口。")
    st.stop()

# 合并为宽表
df_close = pd.DataFrame(price_dict)
df_close = df_close.dropna(how='all') # 去除全NAN的行
df_close = df_close.ffill() # 前向填充停牌数据

# ================= 3. 计算动量因子 =================
# 计算过去N天的收益率
momentum = df_close / df_close.shift(lookback_days) - 1

# ================= 4. 策略回测逻辑 =================
# 这里为了简化，采用每日检查逻辑，但只在每周五调仓
# 计算所有ETF的每日收益率
daily_returns = df_close.pct_change()

portfolio_returns = []
current_holdings = []

for date in momentum.index:
    # 遇到周五（weekday == 4）才进行调仓逻辑
    if date.weekday() == 4:
        today_momentum = momentum.loc[date].dropna()
        if len(today_momentum) > 0:
            # 排序
            sorted_mom = today_momentum.sort_values(ascending=False)

            # 过滤负动量
            valid_etfs = sorted_mom[sorted_mom > 0]

            if len(valid_etfs) == 0:
                current_holdings = [] # 全部为负，空仓
            else:
                current_holdings = valid_etfs.head(top_n).index.tolist()

    # 计算当天组合收益 (如果是周五调仓，下周一才生效；这里简化处理，假设周一开盘执行，当日收益算作次日)
    if current_holdings and date in daily_returns.index:
        # 等权持有
        day_ret = daily_returns.loc[date, current_holdings].mean()
        portfolio_returns.append(day_ret)
    else:
        portfolio_returns.append(0) # 空仓收益为0

strategy_returns = pd.Series(portfolio_returns, index=momentum.index)

# 基准收益（以沪深300为基准）
benchmark_returns = daily_returns['510300'] if '510300' in daily_returns else pd.Series(0, index=momentum.index)

# ================= 5. 计算净值与指标 =================
def calc_cumulative_returns(returns):
    return (1 + returns).cumprod() - 1

cum_strategy = calc_cumulative_returns(strategy_returns)
cum_benchmark = calc_cumulative_returns(benchmark_returns)

# 核心指标
total_ret_strategy = cum_strategy.iloc[-1]
total_ret_bench = cum_benchmark.iloc[-1]
annual_ret = (1 + total_ret_strategy) ** (252 / len(strategy_returns)) - 1

# 最大回撤
roll_max = (1 + cum_strategy).cummax()
drawdown = (1 + cum_strategy) / roll_max - 1
max_drawdown = drawdown.min()

# 夏普比率 (无风险利率按 2% 算)
risk_free_rate = 0.02
excess_daily = strategy_returns - risk_free_rate / 252
sharpe = (excess_daily.mean() / excess_daily.std()) * np.sqrt(252) if excess_daily.std() != 0 else 0

# ================= 6. Streamlit 界面展示 =================
st.subheader("📊 绩效看板")
col1, col2, col3, col4 = st.columns(4)
col1.metric("策略累计收益", f"{total_ret_strategy:.2%}")
col2.metric("年化收益率", f"{annual_ret:.2%}")
col3.metric("最大回撤", f"{max_drawdown:.2%}")
col4.metric("夏普比率", f"{sharpe:.2f}")

st.subheader("📈 净值走势对比")
# 合并净值数据用于绘图
chart_data = pd.DataFrame({
    '策略净值': (1 + cum_strategy),
    '沪深300基准': (1 + cum_benchmark)
})
st.line_chart(chart_data)

# ================= 7. 最新信号提示 =================
st.subheader("🎯 本周最新调仓信号")
latest_date = momentum.index[-1]
latest_mom = momentum.loc[latest_date].sort_values(ascending=False)

signal_df = pd.DataFrame({
    'ETF代码': latest_mom.index,
    'ETF名称': [etf_pool.get(c, "") for c in latest_mom.index],
    '20日动量': latest_mom.values
})
signal_df['20日动量'] = signal_df['20日动量'].apply(lambda x: f"{x:.2%}")

# 判断最新持仓
valid_latest = latest_mom[latest_mom > 0]
if len(valid_latest) == 0:
    st.error("⚠️ 当前所有ETF动量均为负值，策略信号：**空仓**")
else:
    final_holdings = valid_latest.head(top_n).index.tolist()
    st.success(f"✅ 策略信号：建议买入 **{[etf_pool.get(c) for c in final_holdings]}**")
    st.dataframe(signal_df.style.highlight_max(subset=['20日动量'], color='lightgreen'))

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
