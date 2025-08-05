from   utils_common import llm_call, llm_call_rephrase
from   textwrap import dedent
import re
import pandas as pd
import ast

def classify_query(query):
    system_instruction=dedent("""You are tasked with classifying the given query to decide whether it belongs to the "CPI" class, the "GDP" class, the "IIP" class, the "MSME" class, or "Out of domain". Note that there can only be these five options. You proceed using the following hints:
            1. Analyze the provided query for key entities. 
                a. Entities such as inflation, CPI, price index, sale, wholesale, consumer, or consumption belong to the "CPI" class.
                b. The following files comprise the CPI datasets: [cpi_inflation_data,consumer_price_index_CPI_for_agricultural_and_rural_labourers,city_wise_housing_price_indices,whole_sale_price_index_WPI_financial_year_wise,cpi_worker_data,whole_sale_price_index_WPI_calendar_wise]. Any query that can be answered with these data sets should be classified as "CPI".
                c. Entities such as GDP, gross domestic product, GSDP, GVA, NDP, NNI, GNI, capital formation, expenditure components, subsidies by state, and per capita values should belong to the "GDP" class. 
                d. The following files comprise the GDP datasets: [annual_estimate_gdp_crore,annual_estimate_gdp_growth_rate,gross_state_value,key_aggregates_of_national_accounts,per_capita_income_product_final_consumption,provisional_estimateso_gdp_macro_economic_aggregates,quaterly_estimates_of_expenditure_components_gdp,quaterly_estimates_of_gdp]. Any query that can be answered with these data sets should be classified as "GDP".
                e. Entities such as IIP, industrial output, industrial production, mining, manufacturing, electricity, motor vehicles, or other industries should be classified as "IIP".
                f. The following files comprise the IIP datasets: [iip_annual_data,iip_monthly_data]. Any query that can be answered with these data sets should be classified as "IIP".
                g. Entities such as MSME, Micro, Medium and Small Enterprises, credit growth rate, exchange rate, "GDP for MSME", "Nifty SME", Food credit, Non Food Credit, Gross Bank Credit, GDP Contribution by MSME across regions and sectors,economic share of MSME across regions and sectors should be classified as "MSME".
                h. The following files comprise the MSME datasets: [`msme_gbc_food_non_food_view`,`msme_definitions_by_sector`,`msme_gbc_non_food_dtl_view',`msme_gdp_by_region_view',`msme_gdp_by_sector_view',`msme_share_by_region_view',`msme_share_by_sector_view','msme_industry_view','msme_priority_sector_view','nifty_sme_index_daily_values','msme_udyam_registrations_by_state']. Any query that can be answered with these data sets should be classified as "MSME".
            2. Any query that belongs to none of [CPI, GDP, IIP, MSME] should be classified "Out of domain". Examples of out of domain queries include those about MSME (Medium and small businesses), general queries about the economy, queries about government policies, queries about upcoming challenges, and queries unrelated to finance. All of these should be marked "Out of domain". 
            3. EXTREMELY IMPORTANT: Based on the above description, respond ONLY with one of the classes from the following list:
                    [CPI, GDP, IIP, MSME, Out of domain]
                    DO NOT include any reasoning traces or other text apart from the class selected from the above list.
            """)
    query_class = llm_call(system_instruction, query).strip()
    return query_class

def file_selector_CPI(query):
    system_instruction=dedent(f"""you are tasked with identifying the file which contains required data, based on the query: {query}.
                    you must pick one file name only from the following list:
                    1. cpi_inflation_data: This table contains Consumer Price Index (CPI) and inflation data, categorized by year, month, state, sector (Combined, Rural, Urban), group, and sub-group. It includes inflation trends across categories like food, housing, transport, education, and healthcare. This table covers data for year 2017 to 2025. Do NOT choose this table when "workers" or "labourers" are mentioned.
                    2. consumer_price_index_cpi_for_agricultural_and_rural_labourers: this table covers data for year 2024. it should be used only when "agriculture labour" or "rural labour" is mentioned. DO NOT use this file unless "labour" is specifically mentioned.
                    3. city_wise_housing_price_indices: this table covers data for year 2014 to 2024. it should be used only when housing prices are mentioned.
                    4. whole_sale_price_index_wpi_financial_year_wise: this table covers data for year 2012 to 2023. it should be used only when wpi or wholesale prices are mentioned in financial year or fy format.It should be used only when WPI or wholesale prices are mentioned using a financial year format (e.g., "2022-2023", "2023-24", or "FY 2023").
                    5. cpi_worker_data: this table covers data for year 2011 to 2023. it should be used only when WORKERS are mentioned. It contains data about Industrial Workers, Rural Labour, Urban Labour, and Agricultural Labour. DO NOT select this table if query is about  cpi index for food/non-food workers.
                    6. whole_sale_price_index_wpi_calendar_wise: this table covers data for year 2013 to 2023.It should be used only when WPI or wholesale prices are mentioned using a calendar year format (e.g., "2023", "in 2022", or just a single year number).
                    7. cpi_food_worker_data: this table covers CPI data categorized by Food and Non-Food ,specifically for industrial workers. Use this when the focus is on industrial worker ( CPI-IW) Food/Non-Food data.Do not use this table for queries related to CPI for Agricultural and Rural Labourers/workers.
                    8. none_of_these: for any queries which are unrelated to inflation. for example, queries regarding gdp, iip, msme would fall under the "none" category. queries regarding the general state of the economy, government policies, and upcoming challenges also fall under the none_of_these category.
                    
                    ## Consider the list above, and respond ONLY with one of the file names from the following list:
                    [cpi_inflation_data,consumer_price_index_cpi_for_agricultural_and_rural_labourers,city_wise_housing_price_indices,whole_sale_price_index_wpi_financial_year_wise,cpi_worker_data,whole_sale_price_index_wpi_calendar_wise,cpi_food_worker_data,none_of_these]
                    do not include any reasoning traces or other text apart from the file name selected from the above list.
            """)
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file

def file_selector_GDP(query):
    system_instruction=dedent(f"""You are tasked with identifying the file which contains required data, based on the query: {query}.
You MUST pick one file name ONLY from the following list.
For indentifying understand the table names and what sectors/components these tables cover.
Go through entire list and key-words then select the most relevant table.
DO NOT USE annual_estimate_gdp_crore table to answer provisional data related queries.
IMPORTANT RULE:
For any queries LONGER THAN 2 years duration, pick ANNUAL or YEARLY tables where available.


1. india_fy_gdp_view : Aggregated national-level GDP metrics by financial year. Captures GDP and GVA growth at constant and current prices. Any India level summarization needs for GDP across years, trending and growth rate can be done from this table.

2. india_fy_gdp_components_view : Break-up of GDP by economic components (e.g., PFCE, GFCE,GFCF,CIS,VALUABLES, DISCREPANCIES etc) for each FINANCIAL YEAR at INDIA or NATIONAL level. Supports analysis related to components of GDP at year level and share analysis.

3. india_fy_national_income_view : National income accounting components both for Gross and Net Income levels with growth rates at both constant and current prices, financial year-wise.

4. gdp_actuals_summary : Use this table when the query is about yearly state-level economic performance, specifically when it refers to state domestic product (GSDP) or gross value added (GSVA) or growth rates of different states. Note that this data contains ANNUAL information. Always use this table when specific industrial sectors in states are not mentioned, e.g. "what is the GDP of Maharashtra last year". "Top 3 states by GDP". "States driving India's economic growth." "Population used for GSDP calculation".

5. gdp_actuals_details : Use this table when the query is about sectors within states. These components include sectors such as electricity, construction, financial services, transport, etc. Do NOT use this table when no sector is mentioned in the query.

6. per_capita_income_product_final_consumption : This table contains per capita estimates of key economic indicators in ₹ (Indian Rupees) or growth rate (%) for various years at India Level, along with the population used for those calculations. The indicators relate to income and consumption at both current and constant prices.these measures are ["Per Capita GDP","Per Capita GNI","Per Capita NNI","Per Capita GNDI","Per Capita PFCE" , "Percentage change over previous year at constant (2011-12) prices"]

7. quaterly_estimates_of_gdp : This table provides quarterly performance estimates/growth_value of sectors such as: Agriculture, Manufacturing, Services, etc. Growth rates for Trade, Hotels, Transport, Exports, Private Consumption, etc.

8. annual_gdp_estimates : The annual_gdp_estimates table consolidates India’s annual GDP data by combining actual estimates (in ₹ crores) and percentage growth rates across various sectors and components. It includes values at both constant and current prices, enabling analysis of both the size and growth of each sector. This unified structure supports answering both "how much" a sector contributed and "how fast" it grew, making it a key resource for comprehensive GDP-related analysis. Use this table when the query asks for the actual GDP value (in ₹ crores).(i.e If the query is about “how much” a sector or economic indicator contributed to GDP (in ₹ crores) — use this table.). Do not use this file for provisional data. These component/sector includes Trade, Hotels, Transport, Manufacturing, Construction, PFCE, Imports of goods and services, GNI, Per capita income.

**REMEMBER RULES**
- All questions about "top k states" should go to gdp_actuals_summary
- All questions about "states driving economic growth" should go to gdp_actuals_summary

## Table Selection Examples, **Few-shot for india_fy_gdp_view**
Select file: india_fy_gdp_view for queries like those listed below-
Query1: "Compare GDP growth at constant vs current prices for the last 5 years." 
Query2: "What was India's GDP and its growth rate in the most recent year?"
Query3: "How has the GDP growth rate at constant prices evolved over the last 10 years?"


## Table Selection Examples, **Few-shot for india_fy_gdp_components_view**
Select file: india_fy_gdp_components_view for queries like those listed below-
Query: "Give the value of PFCE for the years 2020-21 to 2024-25."  
Query: "How has capital formation (GFCF) changed over the past decade?"  
Query: "Compare the growth of imports and exports of goods and services in the latest year."  
Query: "Show me the trend of GFCE in constant prices over the last 5 years."  
Query: "Which GDP component had the highest growth in 2021-22?"  
Query: "What was the contribution of CIS and VALUABLES to the GDP in 2023-24?"  


## Table Selection Examples **Few-shot for india_fy_national_income_view**
Select file: india_fy_national_income_view for queries like those listed below-
Query: "Show me the Net National Income from 2014-15 to 2020-21."  
Query: "Compare the growth rate of Gross National Income over the past 5 years."  
Query: "What was the GNI and NNI in the year 2023-24?"  
Query: "How did the Net National Income at constant prices trend over the last decade?"  
Query: "Give me the latest Gross National Income in current prices."  
Query: "When did GNI show negative growth at constant prices?"  


## Table Selection Examples **Few-shot for gdp_actuals_summary**
Select file: gdp_actuals_summary for queries like those listed below-
Query: "What was the GSDP of Maharashtra last year?"  
Query: "Top 5 states by GSDP in the most recent year."  
Query: "Show the year-on-year GSDP growth rate for Kerala from 2018-19 to 2023-24."  
Query: "Compare the GSVA at constant prices for Telangana over the past decade."  
Query: "Which states recorded the highest per-capita GSDP in 2024-25?"  
Query: "States driving India’s economic growth in the latest available year."  
Query: "Population used for GSDP calculation"

## Table Selection Examples (Few-shot for gdp_actuals_details)
Select file: gdp_actuals_details for queries like those listed below-
Query: "What is the contribution of manufacturing to Tamil Nadu's GSDP in 2022-23?"  
Query: "Compare the construction sector output of Gujarat and Maharashtra over the last 5 years."  
Query: "Show the trend of electricity sector growth in West Bengal from 2011-12 to 2024-25."  
Query: "How has the financial services industry performed in Karnataka recently?"  
Query: "What was the value added by agriculture in Bihar during 2020-21?"  
Query: "Give a sector-wise GSDP breakdown for Rajasthan for 2023-24."  
Query: "Compare the trade and repair sector across northeastern states in the most recent year."  
Query: "Subsidies on products for South Indian states"

## Table Selection Examples **Few-shot for per_capita_income_product_final_consumption**
Select file: per_capita_income_product_final_consumption for queries like those listed below-
Query: "What is the current per capita income in India?"  
Query: "Give the per capita GDP growth rate for the last five years."  
Query: "What was the per capita private final consumption expenditure in 2014-15?"  
Query: "How has per capita GNI changed from 2011-12 to 2024-25?"  
Query: "Provide the trend of per capita NNI in constant prices over the last decade."  
Query: "Population used for calculating per capita indicators in 2020-21?"  
Query: "Show per capita GNDI values and growth rates since 2015."  

## Table Selection Examples **Few-shot for quaterly_estimates_of_gdp**
Select file: quaterly_estimates_of_gdp for queries like those listed below-
Query: "What was the GDP of India in Q2 of 2023-24?"  
Query: "Give the quarterly growth rate of manufacturing sector over the last year."  
Query: "How did the tertiary sector perform in Q1 2022-23?"  
Query: "Compare PFCE across all four quarters of 2021-22."  
Query: "Show GVA at basic prices and its growth rate in the last three quarters."  
Query: "Provide quarterly values of the primary sector for 2022-23."  
Query: "What is the trend of quarterly GDP growth in 2023-24?"  

## Table Selection Examples **Few-shot for annual_gdp_estimates**
Select file: annual_gdp_estimates for queries like those listed below-
Query: "How much did the manufacturing sector contribute to GDP last year?"
Query: "Compare contribution of primary vs secondary sectors in 2022."
Query: "What is the total value of PFCE for FY 2021-22?"
Query: "Growth rate of public administration and defense in FY 2020?"
Query: "What was the value of Gross Fixed Capital Formation in 2023?"
Query: "What was the contribution of imports to GDP in 2019-20?"

## Consider the list above, and respond ONLY with one of the file names from the following list:
[india_fy_gdp_view, india_fy_gdp_components_view, india_fy_national_income_view, gdp_actuals_summary, gdp_actuals_details, per_capita_income_product_final_consumption, quaterly_estimates_of_gdp, annual_gdp_estimates, none_of_these]
DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
""")
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file

"""
6. key_aggregates_of_national_accounts : Use this table if the query is about macroeconomic indicators (savings, income, consumption, capital formation, ROW transfers) at the national level .These are ["Gross Saving","GVA at basic prices","Less Imports of goods and services","Import of goods","Less Subsidies on Products","GDP","Export of goods","PFCE","GFCE","Primary income receivable from ROW (net)","Import of services","CIS","GNI","Export of services","Export of goods and services","GNDI","Taxes on Products including import duties","CFC","VALUABLES","GCF","Net Saving","NNDI","Rates","GFCF","Less Import of goods and services","GCF excluding Valuables to GDP","Other current transfers (net) from ROW","GCF to GDP","Gross Saving to GNDI","NNI","PFCE to NNI","Exports of goods and services","NDP","Valuables","Discrepancies"]

7. quaterly_key_aggregates_of_national_accounts : Use this table if the query is about quaterly macroeconomic indicators (savings, income, consumption, capital formation, ROW transfers) at the national level .These are ["Gross Saving","GVA at basic prices","Less Imports of goods and services","Import of goods","Less Subsidies on Products","GDP","Export of goods","PFCE","GFCE","Primary income receivable from ROW (net)","Import of services","CIS","GNI","Export of services","Export of goods and services","GNDI","Taxes on Products including import duties","CFC","VALUABLES","GCF","Net Saving","NNDI","Rates","GFCF","Less Import of goods and services","GCF excluding Valuables to GDP","Other current transfers (net) from ROW","GCF to GDP","Gross Saving to GNDI","NNI","PFCE to NNI","Exports of goods and services","NDP","Valuables","Discrepancies"]

9. provisional_estimateso_gdp_macro_economic_aggregates : This table contains provisional estimates of major macroeconomic aggregates like GDP, GVA, NDP, NNI, GNI, capital formation, expenditure components, and per capita values for the most recent years. These are ["Change in stocks","Change in Stocks","Government Final Consumption Expenditure","Gross Fixed Capital Formation","Gross National Income (GNI)","Per capita NNI","Less Imports","Exports","Private Final Consumption Expenditure","Gross fixed capital formation","Private final consumption expenditure","Gross Domestic Product (GDP)","Discrepancies","Valuables","Net Domestic Product (NDP)","Net National Income (NNI)","Government final consumption expenditure","Gross Value Added (GVA) at basic prices","Per Capita NNI"]

10. quaterly_estimates_of_expenditure_components_gdp : This table gives quarter-wise estimates of expenditure components of GDP for multiple years at India Level, at both:Current Prices (nominal, without inflation adjustment),Constant Prices (real, inflation-adjusted).It includes major components like:Private/Government Final Consumption Expenditure,Gross Fixed Capital Formation,Imports & Exports of goods/services,Change in stocks,Discrepancies,Valuables,GDP. These components are ["Private final consumption expenditure","Gross fixed capital formation","Change in stock","Discrepancies","Valuables","Gross domestic product","Government final consumption expenditure","Less: Imports of goods and services","Exports of goods and services"]
"""

def file_selector_IIP(query):
    system_instruction=dedent(f"""You are tasked with identifying the file which contains required data, based on the query: {query}.
    IMPORTANT RULE:
    1. For any queries LONGER THAN 2 years duration, pick ANNUAL or YEARLY tables where available.
    2. If the query is BROAD or GENERAL (e.g. just "IIP growth", "IIP trends", "overall industrial performance") — choose the corresponding *category_view* file (monthly or yearly), not the subcategory-specific files.
    3. Only pick the non-category-view tables (like iip_monthly or iip_yearly) if the query **specifically mentions** detailed manufacturing categories like "textiles", "tobacco", "machinery", etc.
    
    You MUST pick one file name ONLY from the following list:
    1. iip_yearly: This file contains Index of Industrial Production data on an annual or yearly basis. Use this only when specific manufacturing related subcategories (e.g. tobacco products, textiles, apparel, leather, machinery, chemicals, pharma) are queried.
    2. iip_monthly: This file contains Index of Industrial Production data on a monthly basis. Use this file only if the query mentions the word monthly or if it specifies certain months.
    3. iip_yearly_category_view: This file contains Index of Industrial Production data on an annual or yearly basis. This contains high-level information on the three basic sectors (manufacturing, mining, electricity, general) and is preferred for broad queries.
        - Example: IIP in the last three years
    4. iip_monthly_category_view: This file contains Index of Industrial Production data on a monthly basis. This contains high-level information on the three basic sectors (manufacturing, mining, electricity, general) and is preferred for broad queries.
        - Example: Month on month IIP in the last two years
        - Example: IIP of mining sector in March 2023
    Consider the list above, and respond ONLY with one of the file names from the following list:
    [iip_yearly,iip_monthly,iip_yearly_category_view,iip_monthly_category_view,none_of_these]
    DO NOT include any reasoning traces or other text apart from the file name selected from the above list.
    """)
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file

def file_selector_MSME(query):
    system_instruction=dedent(f"""You are tasked with identifying the file which contains required data, based on the query: {query}.
You MUST pick one file name ONLY from the following list:
1. msme_gbc_food_non_food_view : This table contains Monthly Gross Bank Credit (GBC) outstanding, grouped by category (Food credit / Non-Food credit),the column 'gbc_in_cr' holds the amount of Gross Bank Credit (in crores).
2. msme_definitions_by_sector : This table defines the criteria used to classify MSMEs (Micro, Small, and Medium Enterprises) in different Asian countries based on sector (Manufacturing, Services) and criteria like employees, annual income, or annual turnover.  It specifies the thresholds for each category (Micro, Small, Medium), for international MSME classification comparisons.
3. msme_udyam_registrations_by_state : This table contains data on MSME registrations in India, broken down by state. It includes the number of micro, small, and medium enterprises registered, as well as the total number of Udyam registrations and total MSMEs.
4.msme_gbc_non_food_dtl_view : This table contains Monthly Gross Bank Credit (GBC) outstanding by economic sector for non food credit (e.g., Agriculture, Services,Personal Loans,Industries etc.), the column 'gbc_in_cr' holds the amount of Gross Bank Credit (in crores).
5. nifty_sme_index_daily_values : Use this table when the question is for NIFTY SME INDEX for Nifty related data .   
6.msme_gdp_by_region_view : Use this table when the question involves GDP contribution of MSMEs grouped by region, subregion, or country, it contains GDP contribution of MSMEs by region/subregion and country within Asia
7.msme_gdp_by_sector_view : Use this table when the question involves GDP contribution of MSMEs by sector within a region or country.it contains GDP contribution of MSMEs by sector across Asian countries and regions
8.msme_share_by_region_view : Use this table when the question asks for MSME share in the economy grouped by region, subregion, or country.It contains MSME share percentage in the economy, by region/subregion/country
9.msme_share_by_sector_view : Use this table when the question asks for MSME sharebroken down by sector in any region or country.It contains MSME share percentage by sector (e.g., services, manufacturing) across Asia
10.msme_priority_sector_view : This table contains the gross bank credit, GBC outstanding in crores for the top ten most important and in priority sectors among all non food sectors in the column outstanding_as_on
11.msme_industry_view : This table contains the gross bank credit GBC outstanding in crores for all subgroups of the industry sector such as construction, food processing etc., in the column outstanding_as_on
Consider the list above, and respond ONLY with one of the file names from the following list:
[msme_gbc_food_non_food_view, msme_definitions_by_sector, msme_udyam_registrations_by_state, msme_gbc_non_food_dtl_view, nifty_sme_index_daily_values, msme_gdp_by_region_view, msme_gdp_by_sector_view, msme_share_by_region_view, msme_share_by_sector_view, msme_priority_sector_view, msme_industry_view]   
DO NOT include any reasoning traces or other text apart from the file name selected from the above list.


## Table Selection Examples, **Few-shot for msme_gbc_food_non_food_view**
Select file: msme_gbc_food_non_food_view for queries like those listed below-
Query1: "What was the Food Gross Bank Credit on March 2020?"
Query2: "Compare Non-Food and Food GBC for March 2021."
Query3: "Show the monthly trend of Non-Food GBC for the year 2021."
Query4: "How did Food Credit GBC change from Jan 2020 to Dec 2021?"
Query5: "What is the overall Gross Bank Credit trend over last 5 years?"

## Table Selection Examples, **Few-shot for msme_gbc_non_food_dtl_view**
Select file: msme_gbc_non_food_dtl_view for queries like those listed below-
Query1: "What was the Gross Bank Credit to the Agriculture sector in March 2020?"
Query2: "Show the Personal Loans GBC values across all months of 2021."
Query3: "How much GBC was given to the Services sector in April 2021?"
Query4: "List sector-wise GBC on 27 March 2020."
Query5: "Which sector had the highest GBC in 2021? Provide effective and release dates."

## Table Selection Examples, **Few-shot for msme_gdp_by_region_view**
Select file: msme_gdp_by_region_view for queries like those listed below-
Query1: "What is the MSME GDP in South Asia in 2022?"
Query2: "Compare MSME GDP across Southeast Asia and Pacific Islands."

## Table Selection Examples, **Few-shot for msme_gdp_by_sector_view**
Select file: msme_gdp_by_sector_view for queries like those listed below-
Query1: "What is the MSME GDP in the services sector in Central Asia?"
Query2: "GDP trend for manufacturing in Southeast Asia."

## Table Selection Examples, **Few-shot for msme_share_by_region_view**
Select file: msme_share_by_region_view for queries like those listed below-
Query1: "What is the MSME share in Pacific Islands in 2021?"
Query2: "MSME share trend in South Asia."

## Table Selection Examples, **Few-shot for msme_share_by_sector_view**
Select file: msme_share_by_sector_view for queries like those listed below-
Query1: "MSME share in manufacturing sector in Sri Lanka?"
Query2: "Which sector had the highest MSME share in Southeast Asia in 2020?"

## Table Selection Examples, **Few-shot for msme_industry_view**
Select file: msme_industry_view for queries like those listed below-
Query1: "Give the textile industry's outstanding value for the years of 2020 and 2021."

## Table Selection Examples, **Few-shot for msme_priority_sector_view**
Select file: msme_priority_sector_view for queries like those listed below-
Query1: "Give the priority sectors outstanding value for the years of 2020 and 2021."
            """)
    selected_file = llm_call(system_instruction, query).strip()
    return selected_file

def rephrase_for_table(query, schema, context, table_name):
    instructions = f"""
    Given the following table schema: {schema} and the context: {context} for the table {table_name}, can you rephrase the provided query to make it easier for an SQL agent to pull the right data?
    Try to specify values for ALL text columns in the schema that are of "text" type using the provided context, AS LONG AS they are not specified by the query itself. When specifying these values, keep the query in mind and try to use generic values such as Combined, *, General, All India, etc. 
    REMEMBER that these values are to be specified only if they do not clash with the query.
    Do not specify months if the query asks for the whole year. If a date range is specified, make sure you pull ALL the data between those dates.
    
    There are two types of query that you should be able to handle. These are specified below, with logical reasoning:
        1. Time-related queries: These are queries where you are asked about a certain quantity over a certain time period.
           - In this case, fix the categorical columns as far as possible and then set date / year limits on the query.
           - Example: Query --> CPI for vegetables in Karnataka, June 2021 to June 2022
                      SQL query --> SELECT * FROM {table_name} WHERE state = 'Karnataka' AND year >= 2021 AND year <= 2022 AND group_name = 'Food and Beverages' AND sub_group_name = 'Vegetables' AND sector = 'Combined' AND data_source = 'Price Statistics Division, MoSPI' LIMIT 500;
           REMEMBER: Do not set month filters in the SQL query, to avoid confusion.
    
        2. Comparative queries: These are queries that ask for "top k" sort of quantities.
            - For example, this type of query about "top 5 states" which should resolve to top 5 states by GDP, or "top 3 categories" where it should resolve to categories of products.
            - Here, set the time to a specific entry, which should be the latest available in the data.
            - Then, pull all states, categories, etc. that need to be compared without setting any limits on the data.
            - Example: Query --> Top 5 states by GSDP, latest data before June 2025
                    SQL query --> SELECT * FROM {table_name} WHERE year = '2024-25' ORDER BY year DESC
            - Example: Query --> Top 3 categories of inflation, latest data before June 2025
                    SQL query --> SELECT * FROM {table_name} WHERE year = '2025' and month_numeric = '5' AND state = 'All India' AND sector = 'Combined' ORDER BY inflation_rate DESC LIMIT 3;
        
    ** VERY IMPORTANT: **
    If only one year is specified, DO NOT specify month, month_numeric, or quarter in the year.
    NEVER set the value for data release date or the data source in your query.
    Output your response as a valid SQL query.
        query --> Growth of Maharashtra in the last 10 years
        SQL query --> SELECT * FROM {table_name} WHERE state = 'Maharashtra' AND year >= '2015' AND year <= '2025' AND gross_state_value_added_at = 'current price' AND sector = 'Gross State Domestic Product' AND value_unit = 'lakhs' LIMIT 500;
        
        query --> Growth in electricity production from June 2020 to June 2022
        SQL query --> SELECT * FROM {table_name} WHERE year >= '2020-21' AND year <= '2022-23' AND sector_type = 'Sectoral' AND category = 'Electricity' AND sub_category = '*' LIMIT 500;
    """
    rephrased_for_table = llm_call(instructions, query)
    return rephrased_for_table

def identify_generic_columns(schema):
    try:
        #print("Received schema:" )
        #print(schema)
        lines = ast.literal_eval(schema)
        col_list = []
        for line in lines:
            if "Schema" not in line:
                # Use ast.literal_eval to safely parse the tuple
                # Optionally, convert to dict for clarity
                col_dict = {
                    'column_name': line[0],
                    'data_type': line[1],
                    'nullable': line[2],
                    'default': line[3]
                }
                #print("Found tuple: " + str(col_dict))
                if ("id" not in col_dict['column_name']) and \
                   ("year" not in col_dict['column_name']) and \
                   ("month" not in col_dict['column_name']) and \
                   ("data" not in col_dict['column_name']) and \
                   ("date" not in col_dict['column_name']) and \
                   ("released_on" not in col_dict['column_name']) and \
                   ("updated_on" not in col_dict['column_name']) and \
                   (("character varying" in col_dict['data_type']) or ("text" in col_dict['data_type'])):
                       col_list.append(col_dict['column_name'])
                       #col_list.append("WHERE " + col_dict["column_name"] + " = ")
    except:
        col_list = []
    return col_list

def alternative_generate_sql_query(query, schema, context, table_name, last_error="N/A"):
    generic_cols = identify_generic_columns(schema)
    instructions = dedent(f"""
        Here is the schema and a set of sample rows for a table in a postgres SQL database:
            {context}
        
        Task: You have to generate a single VALID SQL QUERY to answer the following natural language query: {query}
        
        Rules:
            1. The table name is {table_name}. This must be used in the FROM section of the query.
            2. We will always pull all columns, so always use SELECT *
            3. Set valid date ranges for all queries. Use "date_stamp" if available, else use "year" or "years".
            4. Do NOT set any conditions using "month" or "month_numeric".
            5. Look for categories such as "General", "Combined", "*" and set them wherever they do not clash with the query {query}.
            6. Try to set specific values for the following columns: {generic_cols} as long as they are not already specified in the query.
            7. DO NOT set any values in the SELECT condition. Use your value setting in relevant columns ONLY in the WHERE clause.
                Example:
                    ALLOWED: SELECT column1, column2, state, year FROM {table_name} WHERE year = 2025 AND state = 'All India'
                    NOT ALLOWED: SELECT column1, column2, year, 'All India' as state FROM {table_name} WHERE year = 2025
                          """)
    if last_error != "N/A":
        instructions += dedent(f"""EXTREMELY IMPORTANT: Keep in mind that your last attempt returned the error: {last_error}
        """)
    instructions += dedent("Give your output ONLY as a valid SQL query, with no other additional text or thinking traces.")
    #print(instructions)
    sql_query = llm_call(instructions, query).strip()
    return sql_query, query

def generate_sql_query(query, schema, context, table_name, last_error="N/A"):
    #if "resulted in no data" not in last_error:
        #columns = str(identify_generic_columns(query, schema))
        #query = str(query) + '''\nEnsure you set the following columns to generic values: ''' + columns)
    if last_error == "N/A":
        sql_query = rephrase_for_table(query, schema, context, table_name)      
        return sql_query, query
          
    instructions = dedent(f"""Given the following table context for {table_name}: {context}\nCan you generate a valid SQL query to get the contents for the natural language query attached below? Be very specific and make sure you output ONLY the SQL query as a string without any other text. Remember to pull all the informative columns in the table, and not just the requested values.
        **Some simple hints to use**
        - Unless the query specifies "data for all categories" or "data for all states", you may choose to pull data for All India, General category, Combined sector, * sub-category, etc. You will get some hints from the sample rows.
        - If the query does specify all categories or a comparison between two categories, states, commodities, etc., then make sure you pull data for all the required fields.
        - Does the provided schema contain any columns called "group", "subgroup", "sector", "state", or similar categorical identifiers? Be sure to set such columns to generic values UNLESS they have been specified in the query.
        - Do the sample rows contain any entries with keywords such as "general", "combined", "All India", or "*"? Using the provided schema, set such values for the relevant columns in the output query, UNLESS the corresponding columns have already been specified. DO NOT specify values for data release dates.
        - When handling time period queries, ensure you pull data for the ENTIRE time period and not for specific months only.
        Example: If the query is "Inflation in food and beverages from May 2023 to May 2025", then use "year >= 2023 AND year >= 2025" and DO NOT set "month_numeric = 5"
        - If you are handling a previous error "SQL query pulled too many rows", then you MUST use the suggested columns and set them to generic values such as "*", "Combined", etc.
        - DO NOT set any values in the SELECT condition. Use your value setting in relevant columns ONLY in the WHERE clause.
        Example:
            ALLOWED: SELECT column1, column2, state, year FROM {table_name} WHERE year = 2025 AND state = 'All India'
            NOT ALLOWED: SELECT column1, column2, year, 'All India' as state FROM {table_name} WHERE year = 2025
        VERY IMPORTANT:
        Do not specify months if the query asks for the whole year. If a date range is specified, make sure you pull ALL the data between those dates.
        If only one year is specified, then DO NOT specify a particular month or quarter.
        EXAMPLE:
            query --> CPI for vegetables in Karnataka, 2021
            SQL query --> SELECT * FROM {table_name} WHERE state = 'Karnataka' AND year = 2021 AND group_name = 'Food and Beverages' AND sub_group_name = 'Vegetables' AND sector = 'Combined' AND data_source = 'Price Statistics Division, MoSPI' LIMIT 500;
        """)
    if last_error != "N/A":
        instructions += dedent(f"""EXTREMELY IMPORTANT: Keep in mind that your last attempt returned the error: {last_error}
        """)
    sql_query = llm_call(instructions, query).strip()
    return sql_query, query

def table_citation(selected_file):
    table_list = {
        'whole_sale_price_index_wpi_financial_year_wise': "Ministry of commerce and industry",
        'quaterly_estimates_of_expenditure_components_gdp': "MoSPI",
        'provisional_estimateso_gdp_macro_economic_aggregates': "MoSPI",
        'per_capita_income_product_final_consumption': "MoSPI",
        'key_aggregates_of_national_accounts': "MoSPI",
        'quaterly_estimates_of_gdp': "MoSPI",
        'annual_estimate_gdp_growth_rate': "MoSPI",
        'gross_state_value': "MoSPI",
        'cpi_worker_data': "Ministry of Finance",
        'annual_estimate_gdp_crore': "MoSPI",
        'iip_data': "Economic Statistics Division, MoSPI",
        'city_wise_housing_price_indices': "Ministry of Finance",
        'consumer_price_index_cpi_for_agricultural_and_rural_labourers': "MoSPI",
        'whole_sale_price_index_wpi_calendar_wise': "Ministry of commerce and industry",
        'cpi_inflation_data': "Price Statistics Division, MoSPI",
        'iip_annual_data': "Economic Statistics Division, MoSPI",
        'iip_monthly_data': "Economic Statistics Division, MoSPI",
        'cpi_food_worker_data': "Ministry of Finance",
        'msme_udyam_registrations_by_state': "Ministry of MSME",
        # 'msme_sector_growth_rates': "RBI",
        # 'msme_global_data': "Asian Development Bank / ERDI",
        'msme_definitions_by_sector': "Asian Development Bank",
        'msme_priority_sector_view': "Ministry of MSME",
        'msme_industry_view': "Ministry of MSME",
        'msme_share_by_sector_view': "Ministry of MSME",
        'msme_share_by_region_view': "Ministry of MSME",
        'msme_gdp_by_sector_view': "Ministry of MSME",
        'msme_gdp_by_region_view': "Ministry of MSME",
        'msme_gbc_food_non_food_view': "Ministry of MSME",
        'msme_gbc_non_food_dtl_view': "Ministry of MSME",
        'nifty_sme_index_daily_values': "NSE Indices"}

    try:
        citation = table_list[selected_file]
    except:
        citation = "Unknown"
    return citation

def data_description(headers):
    system_instruction=dedent("""You are given the following condensed description of the data pulled from internal insights. Can you create a short description of the data in a paragraph between 20 and 50 words? If any json format data is present, also include a couple of insights from the data.
            """)
    description = llm_call(system_instruction, headers)
    return description

def rationalize_information(result, headers, query):
    if query == "":
        query = "Summarize the provided information, and state that this summary is being provided because the data size was too large to answer the query precisely."
    system_instruction=dedent(f""" 
                              You are given the following information about {headers}, in json format:
                                  {result}
                              If you are able to answer the query given below with this information, do so. If not, state that a direct answer is not possible but then summarize the data that is provided.
                              IMPORTANT RULES:
                                  1. DO NOT hallucinate any new information, use only the information provided.
                                  2. DO NOT use your own knowledge, use only the information provided.
                                  3. IGNORE information about ID of the data points and the "base year".
                                  4. If possible, provide the information as a markdown table. This table should be comprehensive based on the provided data. Concentrate on newer data rather than older data. DO NOT miss out on including all information related to the time range in the query.
                                  5. If tabular representation is not possible, provide the information as nicely formatted text (paragraph of around 200 words) or bullet points (approximately 10).
                                  6. Be very brief and focus on answering the provided query. Do not provide decorative information. However, include all data relevant to the time range in {query}.
                              """)
    rationalized_info = llm_call(system_instruction, query)
    return rationalized_info

def handle_pandas_response(df, query, orig_query, max_rows, nq):
    df.fillna('', inplace=True) 
    df.to_csv("debug_dataframe.csv")
    headers = ""
    try:
        if len(df) <= 4:
            result = df.to_dict(orient='records')
            headers = data_description(headers + "\nData: " + str(result)).strip()
            return result, headers
        headers = ""
        # Find single-valued columns
        definite_drops = ["id", "data_release_date", "data_updated_date"]
        single_valued_cols = [col for col in df.columns if (col in definite_drops) or ((df[col].nunique(dropna=False) == 1) and (col.lower() != 'year'))]
        
        # Append their values to the caption
        for col in single_valued_cols:
            val = df[col].iloc[0]
            headers += f"{col}: {val} | "
    
        # Drop trailing delimiter if needed
        headers = headers.rstrip(" | ")
    
        # Drop single-valued columns from the dataframe
        df = df.drop(columns=single_valued_cols)
    
        # Define keywords that indicate temporal association
        temporal_keywords = ['year', 'month', 'quarter', 'date', 'day', 'week', 'period', 'time']
        
        # Create a regex pattern from the keywords
        pattern = re.compile('|'.join(temporal_keywords), re.IGNORECASE)
    
        # Find columns with headers matching any of the temporal keywords
        temporal_cols = [col for col in df.columns if pattern.search(col)]
        selected_temporal_cols = {}
    
        for keyword in temporal_keywords:
            # Filter matching columns for this keyword
            matches = [col for col in temporal_cols if keyword in col.lower() and 'data' not in col.lower()]
            
            # Prioritize numeric columns among the matches
            numeric_matches = [col for col in matches if pd.api.types.is_numeric_dtype(df[col])]
            
            if numeric_matches:
                selected_temporal_cols[keyword] = numeric_matches[0]  # Use the first numeric match
            elif matches:
                selected_temporal_cols[keyword] = matches[0]  # Fallback: first non-numeric match
    
        # Get the selected columns
        cols_to_merge = list(selected_temporal_cols.values())
        
        if 'date_stamp' in list(df):
            df['date'] = df['date_stamp'].astype(str)
        else:
            # Merge into a 'date' column
            if len(temporal_cols) == 1 and temporal_cols[0] == "year":
                df['date'] = pd.to_datetime(df['year'].str[:4], format='%Y')
            else:
                df['date'] = df[cols_to_merge].astype(str).agg('-'.join, axis=1)
        # Convert 'date' column to datetime (will produce NaT for unparseable rows)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        has_nat = df['date'].isna().any()
        if has_nat:
            df['date'] = df['year'].str.extract(r'^(\d{4})').astype(int)
            df['date'] = pd.to_datetime(df['date'], format='%Y')
        
        # Sort the DataFrame by the 'date' columnß
        df = df.sort_values(by='date',ascending=False).reset_index(drop=True)
        try:
            check_top_k = llm_call_rephrase("""Consider the query given below. Your task is to identify if this is a query that compares or ranks certain quantities, categories, states, etc. according to some value.
                For example:
                - "Top 5 states GDP"
                - "Top 3 categories by inflation
                - "Best performing sectors in manufacturing"
                - "States with lowest inflation"
                - "States with highest MSME participation"
            ** CLASSIFICATION TASK: YES or NO**
            1. If such comparisons or rankings exists, reply with a single word "YES"  
            2. If such rankings do not exist, for example "inflation of food category in 2024", "GDP of India in the last 3 years", "IIP of mining sector in the last decade", then reply with a single word "NO"
            3. Do not reply with anything apart from YES or NO
            4. Do not include any thinking traces""", orig_query)
            if check_top_k == "YES":
            #if "top" in orig_query.lower():
                print("Identified as a top-k query, retaining only latest date")
                latest_date = df.loc[0, 'date']
                df = df[df['date'] == latest_date]
        except Exception as e:
            print("Could not assess whether query should keep latest date only: " + str(e))
        df = df.drop(columns=["date"])
        nrows = len(df)
        
        if (nrows <= 12) or (nq == 1):
            result = df.to_dict(orient='records')
            headers = data_description(headers + "\nData: " + str(result)).strip()
            return result, headers
        if 12 < nrows < max_rows:
            # We have more than 12 rows. Need some sort of rationalization.
            # headers += f" -- Data contains {nrows} rows, some rationalization will be needed -- "
            result = df.to_dict(orient='records')
            rationalized_info = rationalize_information(result, headers, orig_query + query).strip()
            headers = data_description(headers).strip()
            return {"summarized_info": rationalized_info}, headers
        # Data size has hit maximum limit. Need some sort of rationalization.
        # headers += f" -- Data contains {nrows} rows, some rationalization will be needed -- "
        result = df.to_dict(orient='records')
        rationalized_info = rationalize_information(result, headers, "Too many rows were pulled, but try to answer the following query from the data provided. Ensure that you mention the date period for which this is valid. **You MUST** use the information from the latest available time period for your summary: " + orig_query + query).strip()
        headers = data_description(headers).strip()
        return {"summarized_info": rationalized_info}, headers
    
    except:
        if "date" in list(df):
            df = df.drop(columns=["date"])
        headers = data_description(headers).strip()
        if len(df) > 100:
            df = df.iloc[:100,:]
        return df.to_dict(orient='records'), headers