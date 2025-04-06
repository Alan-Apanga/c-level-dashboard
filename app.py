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




#%%
st.set_page_config(
    page_title="Dashboard for C-Level",
    page_icon=":bar_chart:",
    layout="wide",
    initial_sidebar_state="expanded")

alt.themes.enable("dark")




#%%

path = './data/*.csv'
csv_files = sorted(glob.glob(path))  
dfs = [pd.read_csv(file) for file in csv_files]

#                       store A
df_inventory = dfs[0]
df_items = dfs[1]
df_purchase_orders = dfs[2]
df_sales_orders = dfs[3]


# Convert 'tranDate' to datetime format
df_sales_orders['tranDate'] = pd.to_datetime(df_sales_orders['tranDate'], errors='coerce')
#print("CSV files found:", csv_files)


#%%



with st.sidebar:
    st.title('💻 C-Level Dashboard')
    
    
    # Store selection
    store_list = ['A', 'G', 'J']
    selected_store = st.selectbox('Select a store', store_list)
    
    
    # Load DataFrames based on store selection
    if selected_store == 'A':
        df_inventory = dfs[0]
        df_items = dfs[1]
        df_purchase_orders = dfs[2]
        df_sales_orders = dfs[3]
    elif selected_store == 'G':
        df_inventory = dfs[4]
        df_items = dfs[5]
        df_purchase_orders = dfs[6]
        df_sales_orders = dfs[7]
    elif selected_store == 'J':
        df_inventory = dfs[8]
        df_items = dfs[9]
        df_purchase_orders = dfs[10]
        df_sales_orders = dfs[11]
    
    
    
    # Convert tranDate column to datetime format for filtering
    df_sales_orders['tranDate'] = pd.to_datetime(df_sales_orders['tranDate'], errors='coerce')
    
    
    
    year_list = sorted(df_sales_orders['tranDate'].dt.year.dropna().unique(), reverse=True)

    # Year selection
    selected_year = st.selectbox('Select a year', year_list, index=len(year_list)-1)
    
    
    
    
    # Filter sales orders for the selected year
    df_selected = df_sales_orders[df_sales_orders['tranDate'].dt.year == selected_year]
    
    
    
    # Sort the filtered dataframe by 'tranDate'
    df_selected_sorted = df_selected.sort_values(by="tranDate", ascending=False)
    

    

#%%



def format_number(num):
    if num > 1000000:
        if not num % 1000000:
            return f'{num // 1000000} M'
        return f'{round(num / 1000000, 1)} M'
    return f'{num // 1000} K'






def calculate_inventory_turnover(df_sales, df_inventory, selected_year):
    """
    Calculate the inventory turnover ratio for the selected year.

    Parameters:
    - df_sales_sorted: Sales orders DataFrame (not yet filtered by year and store).
    - df_inventory: Inventory DataFrame.
    - selected_year: The year selected from the sidebar.

    Returns:
    - inventory_turnover_ratio: The computed inventory turnover ratio.
    """
    # Create a copy to avoid modifying the original DataFrame
    df_sales = df_sales.copy()

    # Ensure 'tranDate' is in datetime format
    df_sales['tranDate'] = pd.to_datetime(df_sales['tranDate'], errors='coerce')

    # Ensure necessary columns exist in df_inventory
    if 'quantityOnHand' not in df_inventory.columns:
        raise ValueError("Column 'quantityOnHand' not found in df_inventory.")
    
    if 'averageCost' not in df_inventory.columns:
        raise ValueError("Column 'averageCost' not found in df_inventory.")

    # Ensure 'COGS' exists in df_sales_sorted
    if 'COGS' not in df_sales.columns:
        # Merge sales data with inventory to get 'averageCost'
        df_sales_sorted = df_sales.merge(df_inventory[['itemGuid', 'averageCost']], 
                                                on='itemGuid', 
                                                how='left')
        

        # Compute COGS: Cost of Goods Sold = |qtyBilled| * averageCost
        df_sales_sorted['COGS'] = df_sales_sorted['qtyBilled'].abs() * df_sales_sorted['averageCost']

    # Filter sales orders for the selected year
    df_sales_sorted = df_sales_sorted[df_sales_sorted['tranDate'].dt.year == selected_year]

    # Compute total COGS for the selected year
    COGS_total = df_sales_sorted['COGS'].sum()

    # Compute average inventory (assuming it's calculated from the full inventory dataset)
    avg_inventory = df_inventory['quantityOnHand'].mean()

    # Calculate inventory turnover ratio
    inventory_turnover_ratio = COGS_total / avg_inventory if avg_inventory > 0 else None

    return inventory_turnover_ratio




def make_donut(pct, status):
    
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
    donut = alt.Chart(source).mark_arc(innerRadius=45, cornerRadius=25).encode(
        theta=alt.Theta(field='Percentage', type='quantitative'),
        color=alt.Color(field='Category', type='nominal',
                        scale=alt.Scale(
                            domain=[status, 'Remaining'],
                            range=[main_color, '#eeeeee']  # Use gray for the remaining part
                        ),
                        legend=None)
    ).properties(width=130, height=130)
    
    
    # Add centered text to show the percentage value
    center_text = alt.Chart(pd.DataFrame({'text': [f"{round(pct)}%"]})).mark_text(
        align='center',
        baseline='middle',
        fontSize=32,
        fontWeight=700,
        fontStyle="italic",
        color=main_color
    ).encode(
        text=alt.Text('text:N')
    ).properties(width=130, height=130)
    
    return donut + center_text
        
        
        
        
@st.cache_data       
def get_order_fulfillment(df_purchase, year):
    """
    Calculate fulfillment rate and filter the DataFrame for a specific year.

    Args:
        df_purchase_orders (pd.DataFrame): Purchase orders DataFrame.
        year (int): Year to filter data.

    Returns:
        tuple: (fulfilled_orders, pending_orders)
    """
    # Create a copy to avoid modifying the original DataFrame
    df_filtered = df_purchase.copy()
    
    columns = ['guid', 'itemGuid', 'tranDate', 'qtyOrdered', 'qtyReceived']
    df_filtered = df_filtered[columns].copy()
    
    # Ensure 'tranDate' is in datetime format (only in df_filtered, not modifying original)
    df_filtered['tranDate'] = pd.to_datetime(df_filtered['tranDate'])

    # Replace zero qtyOrdered to avoid division errors
    df_filtered['qtyOrdered'] = df_filtered['qtyOrdered'].replace(0, np.nan)

    # Calculate Fulfillment Rate (avoid division by zero)
    df_filtered['Fulfillment Rate'] = (df_filtered['qtyReceived'] / df_filtered['qtyOrdered']) * 100

    # Drop NaN values caused by division errors (optional but recommended)
    df_filtered = df_filtered.dropna(subset=['Fulfillment Rate'])

    # Filter for the given year
    df_filtered = df_filtered[df_filtered['tranDate'].dt.year == year]

    # Count fulfilled and pending orders
    fulfilled_orders = df_filtered[df_filtered['Fulfillment Rate'] == 100].shape[0]
    pending_orders = df_filtered[df_filtered['Fulfillment Rate'] < 100].shape[0]

    return fulfilled_orders, pending_orders
        
        
@st.cache_data       
def process_gross_margin(df, selected_year):
    """
    Filters the dataframe by the selected year and calculates the Gross Margin (%).
    
    Parameters:
    - df (pd.DataFrame): The sales orders dataframe.
    - selected_year (int): The year to filter the data.
    
    Returns:
    - pd.DataFrame: The modified dataframe with Gross Margin (%) calculated and cleaned.
    """
    # Ensure 'tranDate' is in datetime format
    df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
    
    # Filter by selected year
    df_filtered = df[df['tranDate'].dt.year == selected_year].copy()
    
    
    
    # Merge Sales and Inventory for COGS(Compute Cost of Goods Sold) Calculation
    df_filtered = df_filtered.merge(df_inventory[['itemGuid', 'averageCost']], on='itemGuid', how='left')

    # ASSUMPTION:returns should not generate profit
    df_filtered['COGS'] = df_filtered['qtyBilled'].abs() * df_filtered['averageCost']
    # Compute Gross Profit
    df_filtered['Gross Profit'] = df_filtered['amount'] - df_filtered['COGS']
    
    # Compute Gross Margin (%)
    df_filtered['Gross Margin (%)'] = np.where(
        df_filtered['amount'].abs() < 1,  # Avoid extreme values
        np.nan,
        np.where(
            df_filtered['amount'] < 0,
            (df_filtered['Gross Profit'] / df_filtered['amount'].abs()) * -100,  # Handle refunds
            (df_filtered['Gross Profit'] / df_filtered['amount']) * 100
        )
    )
    
    # Cap outliers using winsorization
    df_filtered['Gross Margin (%)'] = df_filtered['Gross Margin (%)'].clip(lower=-100, upper=100)
    
    # Fill NaN with 0
    df_filtered['Gross Margin (%)'].fillna(0, inplace=True)
    
    return df_filtered
    
        
        
def plot_gross_margin(df):
    """
    Visualizes the Gross Margin (%) distribution using Altair.
    
    Parameters:
    - df (pd.DataFrame): The dataframe containing the Gross Margin (%) column.
    
    Returns:
    - Altair chart object
    """
    chart = alt.Chart(df).transform_density(
        density='Gross Margin (%)',
        as_=['Gross Margin (%)', 'density'],
    ).mark_area(
        opacity=0.5
    ).encode(
        x='Gross Margin (%):Q',
        y='density:Q',
        color=alt.value("#1f77b4")
    ).properties(
        title="",
        width=600,
        height=400
    )
    
    return chart     
        


#                   Revenue Calculation

@st.cache_data
def plot_revenue_trend(df, selected_year):
    """
    Filters sales orders data for the selected year and plots revenue trend using Altair.

    Args:
        df (pd.DataFrame): DataFrame containing sales orders with 'tranDate' and 'amount'.
        selected_year (int): Year to filter data.

    Returns:
        alt.Chart: Altair line chart showing revenue trend over time.
    """
    
    # Ensure 'tranDate' is in datetime format
    df['tranDate'] = pd.to_datetime(df['tranDate'])
    
    # Filter data by selected year
    df_filtered = df[df['tranDate'].dt.year == selected_year]
    
    # Aggregate revenue per day
    revenue_df = df_filtered.groupby('tranDate')['amount'].sum().reset_index()
    
    # Create Altair line chart
    line_chart = (
        alt.Chart(revenue_df)
        .mark_line(point=True)  # Add points to highlight each day's revenue
        .encode(
            x=alt.X('tranDate:T', title='Date', axis=alt.Axis(format='%b %d, %Y', labelAngle=-45)),  # Format x-axis
            y=alt.Y('amount:Q', title='Revenue', scale=alt.Scale(zero=False)),  # Prevent unnecessary zero baseline
            tooltip=['tranDate', 'amount']  # Interactive tooltip
        )
        .properties(title='', width=600, height=400)
    )

    return line_chart

        



def top10_products_by_category(sales_df, product_df, selected_year):
    """
    Find the top 10 products by sales within each product category for a given year.
    
    Args:
        sales_df (pd.DataFrame): Sales orders DataFrame (e.g., dfs[3]) that contains 'itemGuid', 'amount', and 'tranDate'.
        product_df (pd.DataFrame): Product DataFrame (e.g., dfs[1]) that contains product details,
                                   including 'guid' and 'itemCategory'.
        selected_year (int): Year to filter the data.
                                   
    Returns:
        pd.DataFrame: A DataFrame with the top 10 products by total sales for each product category.
    """
    # Ensure 'tranDate' is in datetime format
    sales_df['tranDate'] = pd.to_datetime(sales_df['tranDate'])

    # Filter sales data for the selected year
    sales_filtered = sales_df[sales_df['tranDate'].dt.year == selected_year]

    # Merge sales orders with product details
    product_cols = ['guid', 'itemCategory']  # Add more columns if needed
    merged_df = sales_filtered.merge(product_df[product_cols], left_on='itemGuid', right_on='guid', how='left')
    
    # Group by product and category, summing the sales amount
    grouped = merged_df.groupby(['itemCategory', 'itemGuid'], as_index=False)['amount'].sum()
    
    # Get the top 10 products for each category sorted by sales amount
    top10 = grouped.groupby('itemCategory', group_keys=False).apply(lambda x: x.nlargest(10, 'amount'))
    
    # Sort results by category and descending amount
    top10 = top10.sort_values(['itemCategory', 'amount'], ascending=[True, False]).head(10)
    
    return top10

#%%


@st.cache_data
def calculate_total_revenue(dfs, selected_year):
    """
    Filters specified DataFrames by year and calculates the total revenue.

    Parameters:
    dfs (list): List of DataFrames.
    selected_year (int): The year to filter by.

    Returns:
    float: Total revenue for the selected year across dfs[3], dfs[7], dfs[11].
    """
    total_revenue = 0
    # indices_to_use = [3, 7, 11]

    # for idx in indices_to_use:
    #     df = dfs[idx].copy()
        
    #     # Ensure datetime format
    #     df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
        
    #     # Filter by year
    #     df_filtered = df[df['tranDate'].dt.year == selected_year]
        
    #     # Sum up revenue
    #     revenue_sum = df_filtered['amount'].sum(skipna=True)
    #     total_revenue += revenue_sum
    
    # calculation for one store
    df = dfs.copy()
    # Ensure datetime format
    df['tranDate'] = pd.to_datetime(df['tranDate'], errors='coerce')
    
    # Filter by year
    df_filtered = df[df['tranDate'].dt.year == selected_year]
    
    # Sum up revenue
    revenue_sum = df_filtered['amount'].sum(skipna=True)
    total_revenue += revenue_sum
    
    

    return total_revenue


        
        
        


#%%

# =============================================================================
# #                 Main Page Display(App layout)
# =============================================================================
# st.title("📊 Filtered Sales Orders")

# spacing of the columns respectively
col = st.columns((1.5, 4.5, 2), gap='medium')



with col[0]:
    

    st.markdown("#### <u>KPI</u>", unsafe_allow_html=True)
      
    
    st.markdown("<br>", unsafe_allow_html=True)  # Spacing
       
    total = calculate_total_revenue(df_sales_orders, selected_year)
    
    if total is None or total < 0:
        total_display = '-'  # Prevent errors
        
    elif selected_year > 2021:
        total_display = format_number(total)
        
    else:
        total_display = '-'

    
    st.metric(label="Total Revenue", value=total_display, delta=None )






    st.markdown("<br>", unsafe_allow_html=True)  # Spacing

    
    
    inventory_TO = calculate_inventory_turnover(df_selected_sorted, df_inventory, selected_year)
    
    if inventory_TO is None or inventory_TO < 0:
        turnover_ratio = '-'  # Prevent errors
        
    elif selected_year > 2021:
        turnover_ratio = format_number(inventory_TO)
        
    else:
        turnover_ratio = '-'

    
    st.metric(label="Inventory Turnover", value=turnover_ratio, delta=None )



    st.markdown("<br>", unsafe_allow_html=True)  # Spacing


    #st.markdown('#### Fulfillment Rate')
    # Calculate fulfillment rate as a new column without modifying the original df_sales_orders in place.
    df_purchase_orders['Fulfillment Rate'] = (df_purchase_orders['qtyReceived'] / df_purchase_orders['qtyOrdered']) * 100
    
    
    
    if selected_year > 2021:   

        # Calculate fulfilled and unfulfilled orders
        # fulfilled_orders = df_purchase_orders[df_purchase_orders['Fulfillment Rate'] == 100].shape[0]
        # pending_orders = df_purchase_orders[df_purchase_orders['Fulfillment Rate'] < 100].shape[0]
        # total_orders = fulfilled_orders + pending_orders
        
        fulfilled_orders, pending_orders = get_order_fulfillment(df_purchase_orders, selected_year)
        total_orders = fulfilled_orders + pending_orders
        
    
        
        # Calculate percentages (ensure total_orders > 0)
        fulfilled_pct = (fulfilled_orders / total_orders) * 100 if total_orders > 0 else 0
        pending_pct   = (pending_orders / total_orders) * 100 if total_orders > 0 else 0
    
        
        # Create two donut charts
        donut_chart_fulfilled_pct = make_donut(fulfilled_pct, 'Fulfilled')
        donut_chart_pending_pct   = make_donut(pending_pct, 'Pending')
        
    else:
        fulfilled_pct = 0
        pending_pct = 0
        donut_chart_fulfilled_pct = make_donut(fulfilled_pct, 'Fulfilled')
        donut_chart_pending_pct   = make_donut(pending_pct, 'Pending')
        
        
    
    fulfillment_col = st.columns((0.2, 1, 0.2))
    st.write('Inbound Fulfilled Orders')
    st.altair_chart(donut_chart_fulfilled_pct)
    st.write('Inbound Pending Orders')
    st.altair_chart(donut_chart_pending_pct)
        
    
    

#%%

with col[1]:
    st.markdown(f"<h4 style='text-align: center;'>Gross Margin Distribution {selected_year}</h4>", unsafe_allow_html=True)
    
#     st.write(f"Data for Store **{selected_store}** and Year **{selected_year}**")
#     df = pd.to_datetime(df_purchase_orders['tranDate'])
#     df = df[df['tranDate'].dt.year == selected_year]
#     st.dataframe(df)  # Display the sorted DataFrame
    
    if selected_year > 2021:  
        df_margin = process_gross_margin(df_selected_sorted, selected_year)
        margin_distr = plot_gross_margin(df_margin)
        revenue_plot = plot_revenue_trend(df_selected_sorted, selected_year)
        
    else:
        st.error("Please select a valid year from the list in the sidebar.")

    
    st.altair_chart(margin_distr, use_container_width=True)
    st.markdown(f"<h4 style='text-align: center;'>Revenue Trend {selected_year}</h4>", unsafe_allow_html=True)
    st.altair_chart(revenue_plot, use_container_width=True)
    
    
    


with col[2]:
    st.markdown('#### Top 10 Products')
    
    if selected_year > 2021:  
        top10_df = top10_products_by_category(dfs[3], dfs[1], selected_year)
    
    else:
        st.error("Please select a valid year from the list in the sidebar.")
    
    
    

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

    st.markdown("<h4 style='text-align: center;'><u>More Information</u></h4>", unsafe_allow_html=True)
    
    with st.expander('About', expanded=True):
        st.write('''
            - Data: Sales and product data merged based on `Product Key`.
            - :orange[**Top Products**]: Displays the **top 10 best-selling products per category** for the selected year.
            - :orange[**Sales Performance**]: Measured in **total sales amount** per product.
            - :orange[**Filtering**]: Data is filtered for the **selected year**.
        ''')

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
    












