import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
from st_aggrid import AgGrid, GridOptionsBuilder
import math
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
from st_aggrid import AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode


def download_template(data_frame, label, file_name):
    # 将数据框转换为 Excel 格式
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    data_frame.to_excel(writer, index=False)
    writer.close()
    st.download_button(label, data=output, file_name=file_name, mime='application/vnd.ms-excel')

def upload_file():
    df = pd.DataFrame(data=None)
    # 创建一个上传文件组件
    uploaded_file = st.file_uploader("上传参数文件")
    # 检查是否有文件上传
    if uploaded_file is not None:
        # 读取上传的文件内容
        file_contents = uploaded_file.read()
        # 将字节流转换为文件对象
        file_obj = BytesIO(file_contents)
        # 将文件内容转换为数据框
        df = pd.read_excel(file_obj)
    return df

def main_calc(df):
    for i, row in enumerate(df.iterrows()):
        min_supply_chain_cost = float('inf')
        min_O = None
        min_ocean_shipping_fee = None
        min_air_shipping_fee = None
        min_ocean_capital_cost_in_transit = None
        min_air_capital_cost_in_transit = None
        min_ocean_additional_storage_fee = None
        min_ocean_capital_cost_additional = None
        min_ocean_additional_throw_cost = None
        min_outage_loss = None
        min_gross_profit_loss = None

        try:
            _, data = row
            skuid = data['SKU信息']
            L = data['L-长度(cm)']
            W = data['W-宽度(cm)']
            H = data['H-高度(cm)']
            Gw = data['Gw实际重量(kg)']
            Cp = data['Cp采购成本(CNY)']
            sales_time_dimension = data['销量时间维度(天)']
            sold_qty = data['对应销售件数']
            price = data['平均售价(CNY)']
            margin = data['销售平均毛利率']
            R = data['R-业务增长率']
            safety_stock = data['默认安全库存']
            default_o = data['O-默认海运比']
            exchange_rate = data['市场汇率(兑CNY)']
            capital_cost_annual_interest_rate = data['默认资金年度成本']
            W1 = data['海运多备货抛货成本权重系数']
            W2 = data['空海运不稳定性造成的断货损失权重系数']
            K = data['安全库存调整系数']
            Co = data['海运运价(元/KG)']
            Ck = data['空运运价(元/KG)']
            Osf = data['海外仓储费(ft3/月)']
            Lo = data['Lo海运时效(天)']
            Lk = data['Lk空运时效(天)']
            Ro = data['Ro海运稳定性']
            Rk = data['Rk空运稳定性']
            volume_weight = round((L * W * H) / 5000, 3)
            if volume_weight > Gw:
                Cw = volume_weight
            else:
                Cw = Gw
            Se = round(sold_qty / sales_time_dimension, 2)
            Lp = round(sales_time_dimension / sold_qty, 2)
            P = round(price * margin * exchange_rate, 2)
            Ce = round(Osf / 30 / (30.48 * 30.48 * 30.48) * (L * W * H), 5)
            Ci = round(Cp * capital_cost_annual_interest_rate / 365, 5)

            for O in np.arange(0, 1, 0.01):
                initial_transportation_cycle = Lk * (1 - O) + Lo * O
                if Se / (safety_stock / initial_transportation_cycle) > 1:
                    e = 1
                else:
                    e = round(Se / (safety_stock / initial_transportation_cycle), 2)
                suggested_safety_stock = math.ceil(initial_transportation_cycle * Se)
                if safety_stock > suggested_safety_stock:
                    # 库存差异数量
                    q = 0
                else:
                    q = suggested_safety_stock - safety_stock
                Np = round(q / suggested_safety_stock, 2)
                if Np <= 0:
                    Np = float(0)
                elif Np>=1:
                    Np = float(1)
                else:
                    pass
                # 建议安全库存
                suggested_safety_stock = math.ceil(initial_transportation_cycle * Se)

                ocean_shipping_fee = round(Co * Cw * O, 3)
                air_shipping_fee = round(Ck * Cw * (1 - O), 3)
                ocean_capital_cost_in_transit = round(Ci * Lo * O, 5)
                air_capital_cost_in_transit = round(Ci * Lk * (1 - O), 5)
                ocean_additional_storage_fee = round(Ce * (1 + (2 * O - 1) / (1 - O)) * (Lo - Lk), 5)
                ocean_capital_cost_additional = round(Ci * (1 + (2 * O - 1) / (1 - O)) * (1 - e) * Lp, 5)
                ocean_additional_throw_cost = round(Cp * (1 + (2 * O - 1) / (1 - O)) * (1 - e) * W1, 5)
                outage_loss = round(P * e * (O * (1 - Ro) + (1 - O) * (1 - Rk)) * (1 + R) * W2, 5)
                gross_profit_loss = round(P * Np * K, 3)
                supply_chain_cost = round(
                    ocean_shipping_fee + air_shipping_fee + ocean_capital_cost_in_transit + air_capital_cost_in_transit \
                    + ocean_additional_storage_fee + ocean_capital_cost_additional + ocean_additional_throw_cost + outage_loss + gross_profit_loss,
                    2)

                if supply_chain_cost < min_supply_chain_cost:
                    min_supply_chain_cost = supply_chain_cost
                    min_O = O
                    min_ocean_shipping_fee = ocean_shipping_fee
                    min_air_shipping_fee = air_shipping_fee
                    min_ocean_capital_cost_in_transit = ocean_capital_cost_in_transit
                    min_air_capital_cost_in_transit = air_capital_cost_in_transit
                    min_ocean_additional_storage_fee = ocean_additional_storage_fee
                    min_ocean_capital_cost_additional = ocean_capital_cost_additional
                    min_ocean_additional_throw_cost = ocean_additional_throw_cost
                    min_outage_loss = outage_loss
                    min_gross_profit_loss = gross_profit_loss

            df.loc[i, '头程海运'] = min_ocean_shipping_fee
            df.loc[i, '头程空运'] = min_air_shipping_fee
            df.loc[i, '海运在途资金成本'] = min_ocean_capital_cost_in_transit
            df.loc[i, '空运在途资金成本'] = min_air_capital_cost_in_transit
            df.loc[i, '海运多备货尾程存储费'] = min_ocean_additional_storage_fee
            df.loc[i, '抛货库存资金成本'] = min_ocean_capital_cost_additional
            df.loc[i, '抛货库存商品成本'] = min_ocean_additional_throw_cost
            df.loc[i, '空海运不稳定性造成的断货损失'] = min_outage_loss
            df.loc[i, '库存不足造成的毛利损失'] = min_gross_profit_loss
            df.loc[i, '最佳海运比'] = min_O
            df.loc[i, '供应链总成本'] = min_supply_chain_cost
            df.loc[i, '建议安全库存'] = suggested_safety_stock
        except:
            st.error(f"{skuid} 数据计算异常")
            continue
    return df


st.title("SKU供应链总成本试算模型")
custom_style = """
<style>
@keyframes color-cycle {
    0% {color: red;}
    50% {color: green;}
    75% {color: blue;}
    100% {color: red;}
}

.custom-caption {
    font-size: 19px;
    opacity: 0.5;
    margin-top: -15px;
    margin-bottom: 5px;
    font-weight: bold; /* 设置字体为粗体 */;
    animation: color-cycle 8s infinite;
}
</style>
"""
# 使用自定义样式
st.markdown(custom_style, unsafe_allow_html=True)
# 在HTML元素上应用自定义类
# st.markdown('<p class="custom-caption">大数据</p>', unsafe_allow_html=True)

if 'template_df' not in st.session_state:
    st.session_state['template_df'] = pd.DataFrame(data=None)

# 事业部-输入参数默认值
default_skuid = ''
default_L = 43.90
default_W = 15.60
default_H = 15.00
default_Gw = 3.63
default_Cp = 2.05
default_sales_time_dimension = 30
default_sold_qty = 12
default_price = 64.99
default_margin = 0.2
default_R = 0.30
default_safety_stock = 5
default_O = 0.5
default_exchange_rate = 7.2891
default_capital_cost_annual_interest_rate = 0.05

# OPC-输入参数默认值
default_Co = 7.0
default_Ck = 36.0
default_Osf = 5.57
default_Lo = 50
default_Lk = 7
default_Ro = 0.95
default_Rk = 0.98

# 权重系数
default_W1 = 1.00
default_W2 = 1.00
default_K = 1.00


with st.expander("批量计算产品最优海运比&供应链成本：", expanded=False):
    col101, col102, col103, col104 = st.columns([5,3,4,4], gap='large')
    with col101:
        parameter_df = upload_file()
    with col102:
        for _ in range(4):
            st.write('')
        data_frame1 = pd.DataFrame(data=None)
        label1 = "文件上传模板下载"
        file_name1 = "SKU供应链总成本试算参数模板.xlsx"
        download_template(data_frame1, label1, file_name1)
    with col103:
        download_result = st.container()

    if not parameter_df.empty:
        button_clicked = st.button("将文件首行填入参数输入框")
        st.write("上传记录：")
        # 配置 AgGrid 表格选项
        # gb2 = GridOptionsBuilder.from_dataframe(parameter_df)
        # 启用单选功能
        # gb2.configure_selection('single', use_checkbox=True)
        # grid_options2 = gb2.build()
        # 在 Streamlit 中显示 AgGrid 表格
        # grid_return = AgGrid(parameter_df,
        #                      gridOptions=grid_options2,
        #                      enable_enterprise_modules=True,
        #                      height=170)

        # # selected_rows = grid_return["selected_rows"]
        st.dataframe(parameter_df)
        if button_clicked:
            # 事业部-输入参数默认值
            st.session_state['template_df'] = parameter_df.head(1).copy()
            default_skuid = st.session_state['template_df'].loc[0, 'SKU信息']
            default_L = float(st.session_state['template_df'].loc[0, 'L-长度(cm)'])
            default_W = float(st.session_state['template_df'].loc[0, 'W-宽度(cm)'])
            default_H = float(st.session_state['template_df'].loc[0, 'H-高度(cm)'])
            default_Gw = float(st.session_state['template_df'].loc[0, 'Gw实际重量(kg)'])
            default_Cp = float(st.session_state['template_df'].loc[0, 'Cp采购成本(CNY)'])
            default_sales_time_dimension = st.session_state['template_df'].loc[0, '销量时间维度(天)']
            default_sold_qty = st.session_state['template_df'].loc[0, '对应销售件数']
            default_price = float(st.session_state['template_df'].loc[0, '平均售价(CNY)'])
            default_margin = float(st.session_state['template_df'].loc[0, '销售平均毛利率'])
            default_R = float(st.session_state['template_df'].loc[0, 'R-业务增长率'])
            default_safety_stock = st.session_state['template_df'].loc[0, '默认安全库存']
            default_O = float(st.session_state['template_df'].loc[0, 'O-默认海运比'])
            default_exchange_rate = float(st.session_state['template_df'].loc[0, '市场汇率(兑CNY)'])
            default_capital_cost_annual_interest_rate = float(st.session_state['template_df'].loc[0, '默认资金年度成本'])

            # OPC-输入参数默认值
            default_Co = float(st.session_state['template_df'].loc[0, '海运运价(元/KG)'])
            default_Ck = float(st.session_state['template_df'].loc[0, '空运运价(元/KG)'])
            default_Osf = float(st.session_state['template_df'].loc[0, '海外仓储费(ft3/月)'])
            default_Lo = st.session_state['template_df'].loc[0, 'Lo海运时效(天)']
            default_Lk = st.session_state['template_df'].loc[0, 'Lk空运时效(天)']
            default_Ro = float(st.session_state['template_df'].loc[0, 'Ro海运稳定性'])
            default_Rk = float(st.session_state['template_df'].loc[0, 'Rk空运稳定性'])

            # 权重系数
            default_W1 = float(st.session_state['template_df'].loc[0, '海运多备货抛货成本权重系数'])
            default_W2 = float(st.session_state['template_df'].loc[0, '空海运不稳定性造成的断货损失权重系数'])
            default_K = float(st.session_state['template_df'].loc[0, '安全库存调整系数'])

    if not parameter_df.empty:
        st.write("计算结果：")
        df_r = main_calc(parameter_df)
        st.dataframe(df_r)
        with download_result:
            for _ in range(4):
                st.write('')
            label1 = "结果下载"
            file_name1 = "SKU供应链总成本试算结果.xlsx"
            download_template(df_r, label1, file_name1)

    st.markdown(f"***导入模板:***")
    input_template_data = {'SKU信息': ['文本类型', '非必填'],
            'L-长度(cm)': ['保留2位小数', '必填'],
            'W-宽度(cm)': ['保留2位小数', '必填'],
            'H-高度(cm)': ['保留2位小数', '必填'],
            'Gw实际重量(kg)': ['保留3位小数', '必填'],
            'Cp采购成本(CNY)': ['保留2位小数', '必填'],
            '销量时间维度(天)': ['默认值30', '必填'],
            '对应销售件数': ['>0整数', '必填'],
            '平均售价(CNY)': ['保留2位小数', '必填'],
            '销售平均毛利率': ['介于0-1保留2位小数', '必填'],
            'R-业务增长率': ['保留2位小数，允许负', '必填'],
            '默认安全库存': ['>0整数', '必填'],
            'O-默认海运比': ['介于0-1保留2位小数', '必填'],
            '市场汇率(兑CNY)': ['最多4位小数，默认', '7.2891'],
            '默认资金年度成本': ['介于0-1保留2位小数，默认', '0.05'],
            '海运多备货抛货成本权重系数': ['保留2位小数，默认', '1'],
            '空海运不稳定性造成的断货损失权重系数': ['保留2位小数，默认', '1'],
            '安全库存调整系数': ['保留2位小数，默认', '1'],
            '海运运价(元/KG)': ['保留1位小数，正值', '必填'],
            '空运运价(元/KG)': ['保留1位小数，正值', '必填'],
            '海外仓储费(ft3/月)': ['保留2位小数，默认', '5.57'],
            'Lo海运时效(天)': ['正整数', '必填'],
            'Lk空运时效(天)': ['正整数', '必填'],
            'Ro海运稳定性': ['保留2位小数，默认', '0.95'],
            'Rk空运稳定性': ['保留2位小数，默认', '0.98']}
    input_template_df = pd.DataFrame(input_template_data)
    st.dataframe(input_template_df)

    st.markdown(f"***导出模板:***")
    output_template_data = {'SKU信息': ['导入值'],
                            'L-长度(cm)': ['导入值'],
                            'W-宽度(cm)': ['导入值'],
                            'H-高度(cm)': ['导入值'],
                            'Gw实际重量(kg)': ['导入值'],
                            'Cp采购成本(CNY)': ['导入值'],
                            '销量时间维度(天)': ['导入值'],
                            '对应销售件数': ['导入值'],
                            '平均售价(CNY)': ['导入值'],
                            '销售平均毛利率': ['导入值'],
                            'R-业务增长率': ['导入值'],
                            '默认安全库存': ['导入值'],
                            'O-默认海运比': ['导入值'],
                            '市场汇率(兑CNY)': ['导入值'],
                            '默认资金年度成本': ['导入值'],
                            '海运多备货抛货成本权重系数': ['导入值'],
                            '空海运不稳定性造成的断货损失权重系数': ['导入值'],
                            '安全库存调整系数': ['导入值'],
                            '海运运价(元/KG)': ['导入值'],
                            '空运运价(元/KG)': ['导入值'],
                            '海外仓储费(ft3/月)': ['导入值'],
                            'Lo海运时效(天)': ['导入值'],
                            'Lk空运时效(天)': ['导入值'],
                            'Ro海运稳定性': ['导入值'],
                            'Rk空运稳定性': ['导入值'],
                            '头程海运': ['中间值'],
                            '头程空运': ['中间值'],
                            '海运在途资金成本': ['中间值'],
                            '空运在途资金成本': ['中间值'],
                            '海运多备货尾程存储费': ['中间值'],
                            '抛货库存资金成本': ['中间值'],
                            '抛货库存商品成本': ['中间值'],
                            '空海运不稳定性造成的断货损失': ['中间值'],
                            '库存不足造成的毛利损失': ['中间值'],
                            '最佳海运比': ['结果值'],
                            '供应链总成本': ['结果值'],
                            '建议安全库存': ['结果值']
                            }
    output_template_df = pd.DataFrame(output_template_data)
    st.dataframe(output_template_df)

# 参数配置
st.subheader('')
with st.expander("事业部-输入参数", expanded=False):
    col01, col02, col03 = st.columns(3, gap='large')
    with col01:
        st.markdown(f"***基础信息:***")
        skuid = st.text_input("SKU信息:", value=default_skuid, help="非必填")
        L = round(float(st.number_input("\u2605 L-长度(cm):", value=default_L, format="%.2f", help="保留2位小数")), 2)
        W = round(float(st.number_input("\u2605 W-宽度(cm):", value=default_W, format="%.2f", help="保留2位小数")), 2)
        H = round(float(st.number_input("\u2605 H-高度(cm):", value=default_H, format="%.2f", help="保留2位小数")), 2)
        Gw = round(float(st.number_input("\u2605 Gw实际重量(kg):", value=default_Gw, format="%.3f", help="保留3位小数")), 3)
        Cp = round(float(st.number_input("\u2605 Cp采购成本(CNY):", value=default_Cp, format="%.2f", help="保留2位小数")), 2)

    with col02:
        st.markdown(f"***销售信息:***")
        sales_time_dimension = st.number_input("销量时间维度(天)：", value=default_sales_time_dimension)
        sold_qty = st.number_input('\u2605 对应销售件数:', min_value=0, value=default_sold_qty, step=1)
        price = round(float(st.number_input(f"\u2605 平均售价(CNY):", value=default_price, format="%.2f")), 2)
        margin = round(float(st.number_input(f"\u2605 销售平均毛利率:", value=default_margin, format="%.2f")), 2)
        R = round(float(st.number_input(f"\u2605 R-业务增长率:", value=default_R, format="%.2f")), 2)
        safety_stock = st.number_input('\u2605 默认安全库存:', min_value=0, value=default_safety_stock, step=1)

    with col03:
        st.markdown(f"***其他信息:***")
        O = round(float(st.number_input(f"\u2605 O-默认海运比:", value=default_O, format="%.2f")), 2)
        exchange_rate = round(float(st.number_input(f"市场汇率(兑CNY):", value=default_exchange_rate, format="%.4f")), 4)
        capital_cost_annual_interest_rate = round(float(st.number_input(f"默认资金年度成本:", value=default_capital_cost_annual_interest_rate, format="%.2f")), 2)

with st.expander("OPC-输入参数", expanded=False):
    col11, col12, col13 = st.columns(3, gap='large')
    with col11:
        Co = round(float(st.number_input(f"\u2605 Co海运运价(元/KG):", value=default_Co, format="%.1f")), 1)
        Ck = round(float(st.number_input(f"\u2605 Ck空运运价(元/KG):", value=default_Ck, format="%.1f")), 1)
        Osf = round(float(st.number_input(f"海外仓储费(ft3/月):", value=default_Osf, format="%.2f")), 2)

    with col12:
        Lo = st.number_input(f"\u2605 Lo海运时效(天):", min_value=0, value=default_Lo, step=1)
        Lk = st.number_input(f"\u2605 Lk空运时效(天):", min_value=0, value=default_Lk, step=1)

    with col13:
        Ro = round(float(st.number_input(f"Ro海运稳定性:", value=default_Ro, format="%.2f")), 2)
        Rk = round(float(st.number_input(f"Rk空运稳定性:", value=default_Rk, format="%.2f")), 2)

with st.expander("参数检查提醒", expanded=False):
    if 'data_check' not in st.session_state:
        st.session_state.data_check = True
    error_messages = []
    if L <= 0 or np.isnan(L):
        error_messages.append("L-长度要求大于0")
    if W <= 0 or np.isnan(W):
        error_messages.append("W-宽度要求大于0")
    if H <= 0 or np.isnan(H):
        error_messages.append("H-高度要求大于0")
    if Gw <= 0 or np.isnan(Gw):
        error_messages.append("Gw实际重量大于0")
    if Cp <= 0 or np.isnan(Cp):
        error_messages.append("Cp采购成本大于0")
    if not isinstance(sold_qty, int) or sold_qty <= 0 or np.isnan(sold_qty):
        error_messages.append("销售件数要求为正整数")
    if price <= 0 or np.isnan(price):
        error_messages.append("平均售价要求大于0")
    if margin <= 0 or margin > 1 or np.isnan(margin):
        error_messages.append("销售平均毛利率要求介于0-1")
    if np.isnan(R):
        error_messages.append("R-业务增长率要求不为空")
    if not isinstance(safety_stock, int) or safety_stock <= 0 or np.isnan(safety_stock):
        error_messages.append("默认安全库存要求为正整数")
    if O <= 0 or O >= 1 or np.isnan(O):
        error_messages.append("O-默认海运比要求介于0-1")
    if Co <= 0 or np.isnan(Co):
        error_messages.append("Co海运运价要求大于0")
    if Ck <= 0 or np.isnan(Ck):
        error_messages.append("Ck空运运价要求大于0")
    if not isinstance(Lo, int) or Lo <= 0 or np.isnan(Lo):
        error_messages.append("Lo海运时效要求为正整数")
    if not isinstance(Lk, int) or Lk <= 0 or np.isnan(Lk):
        error_messages.append("Lk空运时效要求为正整数")

    num_cols = 4  # 设置列数
    if len(error_messages) > 0:
        st.session_state.data_check = False
        # st.error("**错误：**")
        col_list = st.columns(num_cols)
        for i, message in enumerate(error_messages):
            col_idx = i % num_cols
            with col_list[col_idx]:
                st.error("- " + message)
    else:
        st.session_state.data_check = True

st.subheader('')
if st.session_state.data_check:
    with st.expander("注意: 以下是公式计算的中间值, 仅供参考", expanded=True):
        col11, col12, col13 = st.columns(3, gap='large')
        with col11:
            # 体积重量
            volume_weight = round((L * W * H) / 5000, 3)
            volume_weight = st.number_input("体积重量(kg):", value=volume_weight, format="%.3f", disabled = True)
            # Cw计费重量
            if volume_weight > Gw:
                Cw = volume_weight
            else:
                Cw = Gw
            Cw = st.number_input("Cw计费重量(kg):", value=Cw, format="%.3f", disabled = True)
            # 头程海运费
            ocean_shipping_fee = round(Cw * Co, 3)
            ocean_shipping_fee = st.number_input("头程海运费(CNY):", value=ocean_shipping_fee, format="%.3f", disabled=True)
            # 头程空运费
            air_shipping_fee = round(Cw * Ck,3)
            air_shipping_fee = st.number_input("头程空运费(CNY):", value=air_shipping_fee, format="%.3f", disabled=True)
            # 头程平均运输天数
            initial_transportation_cycle = round(Lk * (1 - O) + Lo * O,3)
            initial_transportation_cycle = st.number_input("头程平均运输天数:", value=initial_transportation_cycle, format="%.3f", disabled=True)

        with col12:
            # Se销售效率
            Se = round(sold_qty / sales_time_dimension, 2)
            Se = st.number_input("Se销售效率:", value=Se, format="%.2f", disabled=True)
            # Lp销售周期
            Lp = round(sales_time_dimension / sold_qty, 2)
            Lp = st.number_input("Lp销售周期:", value=Lp, format="%.2f", disabled=True)
            # P-毛利润
            P = round(price * margin * exchange_rate, 2)
            P = st.number_input("P-毛利润(CNY):", value=P, format="%.2f", disabled=True)
            # e-库存去化率
            if Se / (safety_stock / initial_transportation_cycle) > 1:
                e = float(1)
            else:
                e = round(Se / (safety_stock / initial_transportation_cycle), 2)
            e = st.number_input("e-库存去化率:", value=e, format="%.2f", disabled=True)
            # 建议安全库存
            suggested_safety_stock = math.ceil(initial_transportation_cycle * Se)
            suggested_safety_stock = st.number_input("建议安全库存:", value=suggested_safety_stock, disabled=True)

        with col13:
            # 安全库存评测 & 安全库存差额
            if safety_stock > suggested_safety_stock:
                inventory_tip = "安全库存充足"
                q = 0
            else:
                inventory_tip = "安全库存不足"
                q = suggested_safety_stock - safety_stock

            inventory_tip = st.text_input("安全库存评测:", value=inventory_tip)
            q = st.number_input("安全库存差额:", value=q)
            # Np安全库存断货概率
            Np = round(q / suggested_safety_stock, 2)
            if Np <= 0:
                Np = float(0)
            elif Np>=1:
                Np = float(1)
            else:
                pass
            Np = st.number_input("Np安全库存断货概率:", value=Np, format="%.2f", disabled=True)
            # Ce尾程仓储费(元 / 天)
            Ce = round(Osf / 30 / (30.48*30.48*30.48) * (L*W*H),5)
            Ce = st.number_input("Ce尾程仓储费(元 / 天):", value=Ce, format="%.5f", disabled=True)
            # Ci单件资金成本(元 / 天)
            Ci = round(Cp * capital_cost_annual_interest_rate / 365, 5)
            Ci = st.number_input("库存资金成本Ci:", value=Ci, format="%.5f", disabled=True)

    st.subheader('')

if st.session_state.data_check:
    with st.expander("供应链各项成本结果：", expanded=True):
        st.markdown(f"***权重系数:***")
        col_w1, col_w2, col_w3, col_w4 = st.columns(4, gap='large')
        with col_w1:
            W1 = round(float(st.number_input(f"W1海运多备货抛货成本权重系数:", value=default_W1, format="%.2f")), 2)
        with col_w2:
            W2 = round(float(st.number_input(f"W2空海运不稳定性造成的断货损失权重系数:", value=default_W2, format="%.2f")), 2)
        with col_w3:
            K = round(float(st.number_input(f"K安全库存调整系数:", value=default_K, format="%.2f")), 2)

        # 供应链成本计算演示
        cost_dict1 = {}
        cost_dict2 = {}
        cost_dict1['头程海运'] = "Co*Cw*O"
        ocean_shipping_fee = round(Co*Cw*O, 3)
        cost_dict2['头程海运'] = ocean_shipping_fee
        cost_dict1['头程空运'] = "Ck*Cw*(1-O)"
        air_shipping_fee = round(Ck*Cw*(1-O), 3)
        cost_dict2['头程空运'] = air_shipping_fee
        cost_dict1['海运在途资金成本'] = "Ci*Lo*O"
        ocean_capital_cost_in_transit = round(Ci*Lo*O, 5)
        cost_dict2['海运在途资金成本'] = ocean_capital_cost_in_transit
        cost_dict1['空运在途资金成本'] = "Ci*Lk*(1-O)"
        air_capital_cost_in_transit = round(Ci*Lk*(1-O), 5)
        cost_dict2['空运在途资金成本'] = air_capital_cost_in_transit
        cost_dict1['海运多备货尾程存储费'] = "Ce*(1+(2*O-1)/(1-O))*(Lo-Lk)"
        ocean_additional_storage_fee = round(Ce*(1+(2*O-1)/(1-O))*(Lo-Lk), 5)
        cost_dict2['海运多备货尾程存储费'] = ocean_additional_storage_fee
        cost_dict1['抛货库存资金成本'] = "Ci*(1+(2*O-1)/(1-O))*(1-e)*Lp"
        ocean_capital_cost_additional = round(Ci*(1+(2*O-1)/(1-O))*(1-e)*Lp, 5)
        cost_dict2['抛货库存资金成本'] = ocean_capital_cost_additional
        cost_dict1['海运多备货抛货成本'] = "Cp*(1+(2*O-1)/(1-O))*(1-e)*W1"
        ocean_additional_throw_cost = round(Cp*(1+(2*O-1)/(1-O))*(1-e)*W1, 5)
        cost_dict2['海运多备货抛货成本'] = ocean_additional_throw_cost
        cost_dict1['空海运不稳定性造成的断货损失'] = "P*e*(O*(1-Ro)+(1-O)*(1-Rk))*(1+R)*W2"
        outage_loss = round(P*e*(O*(1-Ro)+(1-O)*(1-Rk))*(1+R)*W2, 5)
        cost_dict2['空海运不稳定性造成的断货损失'] = outage_loss
        cost_dict1['库存不足造成的毛利损失'] = "P*Np*K"
        gross_profit_loss = round(P*Np*K, 3)
        cost_dict2['库存不足造成的毛利损失'] = gross_profit_loss
        cost_dict1['供应链成本'] = ""
        supply_chain_cost = round(ocean_shipping_fee + air_shipping_fee + ocean_capital_cost_in_transit + air_capital_cost_in_transit \
            + ocean_additional_storage_fee + ocean_capital_cost_additional + ocean_additional_throw_cost + outage_loss + gross_profit_loss, 2)
        cost_dict2['供应链成本'] = supply_chain_cost

        cost_df1 = pd.DataFrame([cost_dict1, cost_dict2])
        cost_df1.insert(0, '供应链成本', cost_df1.pop('供应链成本'))
        # 构建 columnDefs 对象
        column_defs = [
            {
                "headerName": "",
                "children": [
                    {
                        "headerName": "供应链成本",
                        "field": "供应链成本",
                        "width": 110
                    }
                ]
            },
            {
                "headerName": "头程成本",
                "cellStyle": {"textAlign": "center"},
                "children": [
                    {
                        "headerName": "海运",
                        "field": "头程海运",
                        "width": 90,
                        "suppressSizeToFit": True
                    },
                    {
                        "headerName": "空运",
                        "field": "头程空运",
                        "width": 100
                    }
                ]
            },
            {
                "headerName": "在途库存资金成本",
                "children": [
                    {
                        "headerName": "海运在途",
                        "field": "海运在途资金成本",
                        "width": 100
                    },
                    {
                        "headerName": "空运在途",
                        "field": "空运在途资金成本",
                        "width": 100,
                    }
                ]
            },
            {
                    "headerName": "尾程存储成本",
                    "children": [
                        {
                            "headerName": "海运多备货尾程存储费",
                            "field": "海运多备货尾程存储费",
                            "width": 190
                        }
                    ]

                },
            {
                "headerName": "抛货资金成本",
                "children": [
                    {
                        "headerName": "抛货库存资金成本",
                        "field": "抛货库存资金成本",
                        "width": 190
                    }
                ]
            },
            {
                "headerName": "抛货成本",
                "children": [
                    {
                        "headerName": "海运多备货抛货成本",
                        "field": "海运多备货抛货成本",
                        "width": 200,
                        "suppressSizeToFit": True
                    }
                ]
            },
            {
                "headerName": "断货损失",
                "children": [
                    {
                        "headerName": "空海运不稳定性造成的断货损失",
                        "field": "空海运不稳定性造成的断货损失",
                        "width": 240,
                        "suppressSizeToFit": True
                    },
                    {
                        "headerName": "库存不足造成的毛利损失",
                        "field": "库存不足造成的毛利损失",
                        "width": 180,
                        "suppressSizeToFit": True
                    }
                ]
            }
        ]
        # 配置 AgGrid 表格选项
        gb = GridOptionsBuilder.from_dataframe(cost_df1)
        gb.configure_grid_options(domLayout='normal')
        grid_options = gb.build()
        grid_options['columnDefs'] = column_defs
        # 在 Streamlit 中显示 AgGrid 表格
        AgGrid(cost_df1, gridOptions=grid_options, height=170)

        scheme_data = {'方案类别': ['成本最优方案', '成本最优方案', '时效最优方案'],
                '运输方式组合': ['空运专线', '海运整柜', ''],
                '时效': [Lk, Lo, ''],
                '成本': [Ck, Co, ''],
                '稳定性': [Rk, Ro, ''],
                '方案说明': ['空运经济方案', '海运经济方案', '']}
        scheme_df = pd.DataFrame(scheme_data)
        st.dataframe(scheme_df)

    if st.session_state.data_check:
        with st.expander("遍历计算各海空比的供应链成本：", expanded=True):
            # 遍历计算各海空比的供应链成本
            col81, col82, col83, col84, col85, col86 = st.columns([1, 1, 1, 1, 1, 1])
            with col81:
                min_ocean_rate = st.number_input("海空比最小值:", value=0, min_value=0, max_value=1)
            with col82:
                max__ocean_rate = st.number_input("海空比最大值:", value=1, min_value=0, max_value=1)
            with col83:
                sep_ocean_rate = st.number_input("步长:", value=0.05, min_value=0.01, max_value=1.00)

            cost_list1 = []
            for O in np.arange(min_ocean_rate, max__ocean_rate, sep_ocean_rate):
                cost_dict3 = {}
                initial_transportation_cycle = Lk * (1 - O) + Lo * O
                if Se / (safety_stock / initial_transportation_cycle) > 1:
                    e = 1
                else:
                    e = round(Se / (safety_stock / initial_transportation_cycle), 2)
                suggested_safety_stock = math.ceil(initial_transportation_cycle * Se)
                if safety_stock > suggested_safety_stock:
                    # 库存差异数量
                    q = 0
                else:
                    q = suggested_safety_stock - safety_stock
                Np = round(q / suggested_safety_stock, 2)
                if Np <= 0:
                    Np = float(0)
                elif Np>=1:
                    Np = float(1)
                else:
                    pass

                ocean_shipping_fee = round(Co*Cw*O,3)
                air_shipping_fee = round(Ck*Cw*(1-O),3)
                ocean_capital_cost_in_transit = round(Ci*Lo*O, 5)
                air_capital_cost_in_transit = round(Ci*Lk*(1-O), 5)
                ocean_additional_storage_fee = round(Ce*(1+(2*O-1)/(1-O))*(Lo-Lk), 5)
                ocean_capital_cost_additional = round(Ci*(1+(2*O-1)/(1-O))*(1-e)*Lp, 5)
                ocean_additional_throw_cost = round(Cp*(1+(2*O-1)/(1-O))*(1-e)*W1, 5)
                outage_loss = round(P*e*(O*(1-Ro)+(1-O)*(1-Rk))*(1+R)*W2, 5)
                gross_profit_loss = round(P*Np*K, 3)
                supply_chain_cost = round(
                    ocean_shipping_fee + air_shipping_fee + ocean_capital_cost_in_transit + air_capital_cost_in_transit \
                    + ocean_additional_storage_fee + ocean_capital_cost_additional + ocean_additional_throw_cost + outage_loss + gross_profit_loss, 2)

                cost_dict3['海运比'] = O
                cost_dict3['供应链成本'] = supply_chain_cost
                cost_list1.append(cost_dict3)

            cost_df2 = pd.DataFrame(cost_list1)
            cost_df2['海运比'] = cost_df2['海运比'].apply(lambda x: f'{x*100:.0f}%')

            col91, col92 = st.columns([1, 2])
            with col91:
                min_cost_empty = st.empty()
                highlighted_df = cost_df2.style.highlight_min(subset=['供应链成本'], axis=0)
                formatted_df = highlighted_df.format({'供应链成本': "{:.2f}"})
                st.dataframe(highlighted_df, width=400)
                min_row = cost_df2.nsmallest(1, '供应链成本').reset_index(drop=True)
                ocean_rate_result = min_row.loc[0, '海运比']
                min_supply_chain_cost_result = min_row.loc[0, '供应链成本']
                # 显示文本
                min_cost_empty.markdown(f"<b>海运比为：</b><font color='red'>{ocean_rate_result}</font> <b>时，供应链成本最小：</b><font color='red'>\
                    {min_supply_chain_cost_result}</font>", unsafe_allow_html=True)
            with col92:
                # 创建示例数据
                x = cost_df2['海运比']
                y = cost_df2['供应链成本']

                # 创建折线图
                fig = go.Figure()

                # 添加折线
                fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='sin(x)'))

                # 找到最小值及其索引
                min_y = np.min(y)
                min_index = np.argmin(y)

                # 设置标题和坐标轴标签
                fig.update_layout(
                    title={
                        'text': "供应链成本变化趋势",
                        'x': 0.5,  # 将 x 设置为 0.5 来使标题居中
                        'xanchor': 'center'  # 使用 'center' 锚点来使标题居中
                    },
                    xaxis_title="海运比",
                    yaxis_title="供应链成本"
                )

                # 显示图表
                st.plotly_chart(fig)
            st.subheader('')


