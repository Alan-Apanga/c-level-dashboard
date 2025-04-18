# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 20:31:33 2025

@author: alann
"""

import pandas as pd
import numpy as np
import glob
import streamlit as st
import altair as alt
from datetime import timedelta





#%%
st.set_page_config(
    page_title="Dashboard for C-Level",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

alt.theme.enable("dark")




#%%

@st.cache_data
def load_data():
    path = './data/*.csv'
    csv_files = sorted(glob.glob(path))  
    dfs = [pd.read_csv(file) for file in csv_files]
    
    return dfs
    
#%%


# Set up input widgets
st.logo(
        image="./images/baywa-ag-seeklogo.png",
        icon_image="./images/baywa-seeklogo.png"
)


#                   Data Loading & Organization
dfs = load_data()

# df_inventory = dfs[0]
# df_items = dfs[1]
# df_purchase_orders = dfs[2]
# df_sales_orders = dfs[3]


# df_sales_orders.info()

# Store mapping: Store A: 0,1,2,3; Store G: 4,5,6,7; Store J: 8,9,10,11
store_map = {'A': (0,1,2,3), 'G': (4,5,6,7), 'J': (8,9,10,11)}
    

#%%

#                       HELPER METHODS

def format_number(num):
    """Format numbers with K, M, or B."""
    abs_num = abs(num)
    if abs_num >= 1_000_000_000:
        return f"{num/1_000_000_000:.1f} B"
    elif abs_num >= 1_000_000:
        return f"{num/1_000_000:.1f} M"
    elif abs_num >= 1_000:
        return f"{num/1_000:.1f} K"
    else:
        return str(num)
    

#%%
    
#                        Retrieve data for selected store(s)
    
    
def get_store_data(dfs, store):
    
    if store == "All":
        
        # Combine each type of file across all stores
        inv_dfs = [dfs[store_map[s][0]] for s in store_map]
        prod_dfs = [dfs[store_map[s][1]] for s in store_map]
        pur_dfs = [dfs[store_map[s][2]] for s in store_map]
        sales_dfs = [dfs[store_map[s][3]] for s in store_map]
        
        
        # Combines a list of DataFrames (here called inv_dfs) into a single DataFrame, 
        # stacking them vertically (row-wise) by default
        df_inventory = pd.concat(inv_dfs, ignore_index=True)
        df_products = pd.concat(prod_dfs, ignore_index=True)
        df_purchase = pd.concat(pur_dfs, ignore_index=True)
        df_sales = pd.concat(sales_dfs, ignore_index=True)
    
    else:
        idx0, idx1, idx2, idx3 = store_map[store]
        df_inventory = dfs[idx0]
        df_products = dfs[idx1]
        df_purchase = dfs[idx2]
        df_sales = dfs[idx3]
    
    return df_inventory, df_products, df_purchase, df_sales
    
#%%

@st.cache_data
def filter_sales(df, start, end):
    df = df.copy()
    df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
    return df[(df['tranDate'] >= start) & (df['tranDate'] <= end)]  



#%%

def compute_delta(df, column):
    # Ensure at least two records exist
    if len(df) < 2:
        return 0, 0
    cur_val = df[column].iloc[-1]
    prev_val = df[column].iloc[-2]
    delta = cur_val - prev_val
    delta_pct = (delta / prev_val) * 100 if prev_val != 0 else 0
    return delta, delta_pct


#%%

# Data aggregation based on time frame
def aggregate_timeframe(df, freq):
    """ 
    freq can be a pandas offset alias like:

    'D' – daily
    
    'W' – weekly
    
    'M' – monthly
    
    'Q' – quarterly
    
    'Y' – yearly
    """
    
    df = df.copy()
    df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
    # When using altair charts you may prefer to have the date as the index
    df = df.set_index('tranDate')
    return df.resample(freq).sum(numeric_only=True) # groups data based on freq datetime index.


def get_daily_data(df):
    return df.set_index('tranDate')

def get_weekly_data(df):
    return aggregate_timeframe(df, 'W-MON')

def get_monthly_data(df):
    return aggregate_timeframe(df, 'M')

def get_quarterly_data(df):
    return aggregate_timeframe(df, 'Q')


#%%

#               KPI calculations

def calculate_total_revenue(df):
    return df['amount'].sum()


def calculate_inventory_turnover(sales_df, inventory_df):
    if 'COGS' not in sales_df.columns:
        # Merge to compute COGS if not already in the dataframe
        sales_merged = sales_df.merge(inventory_df[['itemGuid', 'averageCost']], on='itemGuid', how='left')
        sales_merged['COGS'] = sales_merged['qtyBilled'].abs() * sales_merged['averageCost']
    else:
        sales_merged = sales_df
    total_COGS = sales_merged['COGS'].sum()
    avg_inventory = inventory_df['quantityOnHand'].mean()
    return total_COGS/avg_inventory if avg_inventory > 0 else np.nan


def process_gross_margin(sales_df, inventory_df):
    df = sales_df.copy()

    # Merge cost info
    df = df.merge(inventory_df[['itemGuid', 'averageCost']], on='itemGuid', how='left')

    # Calculate COGS and Gross Profit
    df['COGS'] = df['qtyBilled'].abs() * df['averageCost']
    df['Gross Profit'] = df['amount'] - df['COGS']

    # Avoid divide-by-zero
    total_revenue = df['amount'].sum()
    if abs(total_revenue) < 1e-6:
        return 0.0  # or np.nan depending on how you want to handle zero revenue

    # Weighted Gross Margin (%)
    gross_margin_weighted = (df['Gross Profit'].sum() / total_revenue) * 100

    return round(gross_margin_weighted, 2)


def calculate_fulfillment_breakdown(fulfilled, pending):
    """
    Calculates the percentage breakdown of fulfilled and pending orders.

    If total (fulfilled + pending) is negative, percentages are returned as negative.
    If total is zero, both percentages are returned as 0 to avoid division by zero.

    Args:
        fulfilled (int or float): Number of fulfilled orders.
        pending (int or float): Number of pending orders.

    Returns:
        tuple: (fulfilled_pct, pending_pct)
    """
    total_orders = fulfilled + pending

    if total_orders == 0:
        return 0.0, 0.0

    # Base percentages
    fulfilled_pct = (fulfilled / abs(total_orders)) * 100
    pending_pct = (pending / abs(total_orders)) * 100

    # Preserve sign if total is negative
    if total_orders < 0:
        fulfilled_pct *= -1
        pending_pct *= -1

    return round(fulfilled_pct, 2), round(pending_pct, 2)

#%%
#               Order Fulfillment Functions

@st.cache_data
def get_order_fulfillment(df_purchase, start, end):
    """
    Filter purchase orders for the selected period and compute fulfilled vs pending orders.
    A fulfilled order has a fulfillment rate of 100%.
    """
    df = df_purchase.copy()
    df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
    df = df[(df['tranDate'] >= start) & (df['tranDate'] <= end)]
    
    # Avoid division by zero by replacing zero orders with NaN
    df["qtyOrdered"] = df["qtyOrdered"].replace(0, np.nan)
    df["Fulfillment Rate"] = (df["qtyReceived"] / df["qtyOrdered"]) * 100
    df = df.dropna(subset=["Fulfillment Rate"])
    fulfilled_orders = df[df["Fulfillment Rate"] == 100].shape[0]
    pending_orders   = df[df["Fulfillment Rate"] < 100].shape[0]
    return fulfilled_orders, pending_orders
#%%
def make_donut(pct, status):
    
    
   
    size=130
    inner_radius=45
    corner_radius=25
    text_size=32
    
    
    
    # Choose colors based on the status
    if status == 'Fulfilled':
        main_color = '#2ca02c'  # Green for fulfilled
    elif status == 'Pending':
        main_color = '#d62728'  # Red for pending
    else:
        main_color = '#888888'  # Fallback color
        
    
    # Create a source DataFrame with two parts: the given percentage and the remaining percentage.
    source = pd.DataFrame({
        'Category': [status, 'Remaining'],
        'Percentage': [pct, 100 - pct]
    })
    
    
    # Create the donut chart
    donut = alt.Chart(source).mark_arc(innerRadius=inner_radius, cornerRadius=corner_radius).encode(
        theta=alt.Theta(field='Percentage', type='quantitative'),
        color=alt.Color(field='Category', type='nominal',
                        scale=alt.Scale(
                            domain=[status, 'Remaining'],
                            range=[main_color, '#eeeeee']  # Use gray for the remaining part
                        ),
                        legend=None)
    ).properties(width=size, height=size)
    
    
    # Add centered text to show the percentage value
    center_text = alt.Chart(pd.DataFrame({'text': [f"{round(pct)}%"]})).mark_text(
        align='center',
        baseline='middle',
        fontSize=text_size,
        fontWeight=700,
        fontStyle="italic",
        color=main_color
    ).encode(
        text=alt.Text('text:N')
    ).properties(width=size, height=size)
    
    return donut + center_text



#%%

def display_metric(string, value):
    with st.container(border=True):
        st.metric(string, format_number(value),
                  delta=f"{format_number(delta_rev)} ({delta_rev_pct:+.2f}%)")
            

#%%


def top10_products(sales_df, prod_df):
        sales_df['tranDate'] = pd.to_datetime(sales_df['tranDate'])
        merged = sales_df.merge(prod_df[['guid', 'itemCategory']], left_on='itemGuid', right_on='guid', how='left')
        grouped = merged.groupby(['itemCategory', 'itemGuid'], as_index=False)['amount'].sum()
        grouped['itemGuid'] = grouped['itemGuid'].astype(str)
        top10 = grouped.sort_values("amount", ascending=False).head(10)
        return top10

#%%

# =============================================================================
# #                 Main Page Display(App layout)
# =============================================================================

#%%
with st.sidebar:
    st.title('💻 High-Level Dashboard')
    st.header("⚙️ Settings")
    
    
    # # Store selection
    # store_list = ['A', 'G', 'J']
    # selected_store = st.selectbox('Select a store', store_list)
    
    
    # Select view mode: Combined or individual store
    view_mode = st.radio("Select view mode", options=["All Stores", "Individual Store"])
    if view_mode == "Individual Store":
        selected_store = st.selectbox("Select a store", options=list(store_map.keys()))
    else:
        selected_store = "All"
        
        
    
    # Get the global date range across all sales CSVs (indices 3, 7, 11 for sales)
    def get_global_date_range(dfs):
        dates = []
        for idx in [3, 7, 11]:
            df_temp = dfs[idx].copy()
            df_temp['tranDate'] = pd.to_datetime(df_temp['tranDate'], errors='coerce')
            dates += list(df_temp['tranDate'].dropna())
        return min(dates), max(dates)
    
    global_start, global_end = get_global_date_range(dfs)
    
    
    # Date range selector
    # date_range = st.date_input("Select Date Range",
    #                            value=(global_start, global_end),
    #                            min_value=global_start,
    #                            max_value=global_end)
    
    # start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    
    
    default_start_date = global_end - timedelta(days=365)  # Show a year by default
    default_end_date = global_end
    start = st.date_input("Start date", default_start_date, min_value=global_start, max_value=global_end)
    end = st.date_input("End date", default_end_date, min_value=global_start, max_value=global_end)
    start_date = pd.to_datetime(start)
    end_date = pd.to_datetime(end)
    
    
    
    # Time frame and chart type selection
    time_frame = st.selectbox("Select time frame", ("Daily", "Weekly", "Monthly", "Quarterly"))
    #chart_type = st.selectbox("Select a chart type", ("Bar", "Area"))
    
    
    # For delta computation: define a previous period with the same duration
    period_delta = end_date - start_date # duration of the current period
    prev_start = start_date - period_delta - timedelta(days=1) # Compute the previous period of the same duration
    prev_end = start_date - timedelta(days=1)
    st.markdown(f"**Comparison Period:**<br>{prev_start} to {prev_end}", unsafe_allow_html=True)

    
    
    
#%%
    
df_inventory, df_products, df_purchase, df_sales = get_store_data(dfs, selected_store) 
    
# Filter sales data to current period and previous period for delta
sales_current = filter_sales(df_sales, start_date, end_date)
sales_prev = filter_sales(df_sales, prev_start, prev_end)

#orders_current = filter_sales(df_purchase, start_date, end_date)
    

# Get aggregated view based on time frame
if time_frame == 'Daily':
    sales_current_disp = get_daily_data(sales_current)
    sales_prev_disp = get_daily_data(sales_prev)
elif time_frame == 'Weekly':
    sales_current_disp = get_weekly_data(sales_current)
    sales_prev_disp = get_weekly_data(sales_prev)
elif time_frame == 'Monthly':
    sales_current_disp = get_monthly_data(sales_current)
    sales_prev_disp = get_monthly_data(sales_prev)
elif time_frame == 'Quarterly':
    sales_current_disp = get_quarterly_data(sales_current)
    sales_prev_disp = get_quarterly_data(sales_prev)
    
#%%

# Fulfillment Metrics & Donut Charts for Orders
fulfilled, pending = get_order_fulfillment(df_purchase, start_date, end_date)
fulfilled_pct, pending_pct =  calculate_fulfillment_breakdown(fulfilled, pending)
donut_chart_fulfilled = make_donut(fulfilled_pct, 'Fulfilled')
donut_chart_pending   = make_donut(pending_pct, 'Pending')
    
    

#%%


#               Sales Comparison Chart Across Stores
# Combined Metrics (for both Combined and Individual views)
if selected_store == "All" or view_mode == "Combined All Stores":
    st.markdown("## All Stores Metrics")
    
    total_rev_current = calculate_total_revenue(sales_current)
    total_rev_prev = calculate_total_revenue(sales_prev)
    delta_rev, delta_rev_pct = compute_delta(pd.DataFrame({"amount": [total_rev_prev, total_rev_current]}), "amount")
    
    inv_turnover_current = calculate_inventory_turnover(sales_current, df_inventory)
    inv_turnover_prev = calculate_inventory_turnover(sales_prev, df_inventory)
    delta_inv_df = pd.DataFrame({"inv_turnover": [inv_turnover_prev, inv_turnover_current]}) # temp df
    delta_inv, delta_inv_pct = compute_delta(delta_inv_df, "inv_turnover")
    
    avg_gross_margin_current = process_gross_margin(sales_current, df_inventory)
    avg_gross_margin_prev = process_gross_margin(sales_prev, df_inventory)
    delta_gm_df = pd.DataFrame({"gross_margin": [avg_gross_margin_prev, avg_gross_margin_current]}) # temp df
    delta_gm, _ = compute_delta(delta_gm_df, "gross_margin")
    
    top10_df = top10_products(sales_current, df_products)
    
    
    
    
    kpi_cols = st.columns(3)
    with kpi_cols[0]:
        with st.container(border=True):
            st.metric("Total Revenue", format_number(total_rev_current),
                      delta=f"{format_number(delta_rev)} ({delta_rev_pct:+.2f}%)")
    with kpi_cols[1]:
        with st.container(border=True):
            st.metric("Inventory Turnover", format_number(inv_turnover_current),
                      delta=f"{format_number(delta_inv)} ({delta_inv_pct:+.2f}%)")
    
    with kpi_cols[2]:
        with st.container(border=True):
            st.metric("Avg Gross Margin (%)", f"{avg_gross_margin_current:.1f}%",
                      delta=f"{delta_gm:+.1f}%")
    
    
    
    orders_kpi_cols = st.columns(2)
    with orders_kpi_cols[0]:
        with st.container(border=True):
            st.markdown("**Inbound Fulfilled Orders**")
            st.altair_chart(donut_chart_fulfilled, use_container_width=True)
            
    with orders_kpi_cols[1]:
        with st.container(border=True):
            st.markdown("**Inbound Pending Orders**")
            st.altair_chart(donut_chart_pending, use_container_width=True)
            
          
    
    with st.expander('Top Products', expanded=True):
        #st.markdown("**Top Products**")
        
        
        # Display the dataframe in Streamlit
        st.dataframe(top10_df,
                     column_order=("itemCategory", "itemGuid", "amount"),
                     hide_index=True,
                     width=None,
                     column_config={
                        "itemCategory": st.column_config.TextColumn(
                            "Product Category",
                        ),
                        "itemGuid": st.column_config.TextColumn(
                            "Product ID",
                        ),
                        "amount": st.column_config.ProgressColumn(
                            "Total Sales Amount",
                            format="%.2f",
                            min_value=0,
                            max_value=max(top10_df.amount) if not top10_df.empty else 1,  # Avoid division by zero
                         )} 
                     )
    
    st.markdown("#### Sales Comparison Across Stores")
        
    with st.container(border=True):

        chart_data = pd.DataFrame()
        for s in store_map.keys():
            _, _, _, sales_temp = get_store_data(dfs, s)
            temp = filter_sales(sales_temp, start_date, end_date).copy()
            temp["Store"] = s
            chart_data = pd.concat([chart_data, temp], ignore_index=True)
        
        chart_data['tranDate'] = pd.to_datetime(chart_data['tranDate'], errors='coerce')
        
        # Aggregate total amount by store
        store_sales = chart_data.groupby("Store")["amount"].sum().reset_index()
        
        # Create bar chart
        sales_chart = alt.Chart(store_sales).mark_bar().encode(
            x=alt.X('Store:N', title='Store', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('amount:Q', title='Total Revenue'),
            color='Store:N',
            tooltip=['Store', 'amount']
        ).properties(width=600, height=200)
        
        text = sales_chart.mark_text(
            align='center',
            baseline='bottom',
            dy=-5  # offset
        ).encode(
            text=alt.Text('amount:Q', format='.2s')
        )
        
        final_sales_chart = sales_chart + text
        
        st.altair_chart(final_sales_chart, use_container_width=True)


    
    
    
#%%  
    
#st.markdown('<hr style="border: 1px solid #999;">', unsafe_allow_html=True)

#               Store Level Dashboard (if Individual Store)

if view_mode == "Individual Store":
    st.markdown(f"## Dashboard for Store {selected_store}")

    total_rev_store = calculate_total_revenue(sales_current)
    total_rev_prev_store = calculate_total_revenue(sales_prev)
    delta_store_df = pd.DataFrame({"amount": [total_rev_prev_store, total_rev_store]})
    delta_store, delta_store_pct = compute_delta(delta_store_df , "amount")
    
    inv_turnover_store = calculate_inventory_turnover(sales_current, df_inventory)
    inv_turnover_store_prev = calculate_inventory_turnover(sales_prev, df_inventory)
    delta_store_inv_df = pd.DataFrame({"inv_store_turnover": [inv_turnover_store_prev, inv_turnover_store]}) # temp df
    delta_store_inv, delta_store_inv_pct = compute_delta(delta_store_inv_df, "inv_store_turnover")
    
    
    avg_gross_margin_store = process_gross_margin(sales_current, df_inventory)
    avg_gross_margin_store_prev = process_gross_margin(sales_prev, df_inventory)
    delta_gm_store_df = pd.DataFrame({"gross_margin": [avg_gross_margin_store_prev, avg_gross_margin_store]}) # temp df
    delta_store_gm, _ = compute_delta(delta_gm_store_df, "gross_margin")
    
    top10_store = top10_products(sales_current, df_products)

    # 1st Row
    store_cols = st.columns(3)
    with store_cols[0]:
        with st.container(border=True):
            st.metric("Total Revenue", format_number(total_rev_store),
                          delta=f"{format_number(delta_store)} ({delta_store_pct:+.2f}%)")
            
    
    with store_cols[1]:
        with st.container(border=True):
            st.metric("Inventory Turnover", format_number(inv_turnover_store),
                      delta=f"{format_number(delta_store_inv)} ({delta_store_inv_pct:+.2f}%)")
            
    
    with store_cols[2]:
        with st.container(border=True):
            st.metric("Avg Gross Margin (%)", f"{avg_gross_margin_store:.1f}%",
                      delta=f"{delta_store_gm:+.1f}%")
            
    # 2nd Row
    store_orders_col = st.columns(2)
    with store_orders_col[0]:
        with st.container(border=True):
            st.markdown("**Inbound Fulfilled Orders**")
            st.altair_chart(donut_chart_fulfilled, use_container_width=True)
            
    with store_orders_col[1]:
        with st.container(border=True):
            st.markdown("**Inbound Pending Orders**")
            st.altair_chart(donut_chart_pending, use_container_width=True)
            
            
            
    # 3rd Row
    with st.expander('Top Products', expanded=True):# Display the dataframe in Streamlit
        st.dataframe(top10_store,
                     column_order=("itemCategory", "itemGuid", "amount"),
                     hide_index=True,
                     width=None,
                     column_config={
                        "itemCategory": st.column_config.TextColumn(
                            "Product Category",
                        ),
                        "itemGuid": st.column_config.TextColumn(
                            "Product ID",
                        ),
                        "amount": st.column_config.ProgressColumn(
                            "Total Sales Amount",
                            format="%.2f",
                            min_value=0,
                            max_value=max(top10_store.amount) if not top10_store.empty else 1,  # Avoid division by zero
                         )} 
                     )
    
    
    
    
    st.markdown("### Sales Trend")
    with st.container(border=True):
        sales_trend = sales_current_disp.groupby(sales_current_disp.index)["amount"].sum().reset_index()
        trend_chart = alt.Chart(sales_trend).mark_line(point=True).encode(
            x=alt.X('tranDate:T', title='Date', axis=alt.Axis(format='%b %d', labelAngle=-45)),
            y=alt.Y('amount:Q', title='Revenue'),
            tooltip=['tranDate', 'amount']
        ).properties(width=800, height=400)
        st.altair_chart(trend_chart, use_container_width=True)
    
    
    
    st.markdown('<hr style="border: 1px solid #999;">', unsafe_allow_html=True)
    
    
    
    # DataFrame Info display for associated CSV files
    # st.markdown("### Detailed Data Overview")
    # with st.expander("View DataFrame Info for the store files"):
    #     for label, df_temp in zip(["Inventory", "Products", "Purchase Orders", "Sales Orders"],
    #                               [df_inventory, df_products, df_purchase, df_sales]):
    #         st.write(f"**{label}**")
            
    #         # Use StringIO buffer to capture df.info()
    #         buffer = io.StringIO()
    #         df_temp.info(buf=buffer)
    #         info_str = buffer.getvalue()
            
            
            
    #         st.text(info_str)
    #         st.markdown("---")
    

#%%

# DataFrame display for selected time frame
# with st.expander('Data: Selected Duration', expanded=True):

#     df_display = sales_current_disp.copy()      
#     # Convert problematic object-type columns to string
#     for col in df_display.columns:
#         if df_display[col].dtype == 'object':
#             df_display[col] = df_display[col].astype(str)
#     st.dataframe(df_display)   
        
    
            
#%%





