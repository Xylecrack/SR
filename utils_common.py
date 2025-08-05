#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:08:22 2025

@author: harshad

Common utils for SQL and VEC data
"""
from   dotenv import load_dotenv
import os
import time
from   time import strftime, gmtime
#from   google.genai import types
from   google.genai.types import Tool, GoogleSearch, GenerateContentConfig
from   google import genai
import google.generativeai as gai
from   textwrap import dedent
from   groq import Groq
#from   googlesearch import search

load_dotenv("prod.env")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
#print("Loaded groq key " + GROQ_API_KEY)
gai.configure(api_key=GOOGLE_API_KEY)
client = genai.Client(api_key=GOOGLE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)
model_id = "gemini-2.0-flash"
# Enable Google Search tool
google_search_tool = Tool(google_search=GoogleSearch())

def llm_call_rephrase(system_instruct, contents, tool_list=[]):
        
    prompt = f"{system_instruct}\n\nProvided context: {contents}"
    
    response = groq_client.chat.completions.create(
        model="llama3-70b-8192",  # Example model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,    # Disables randomness
        seed=42,            # Fixed seed for reproducibility
        max_tokens=500,
    )
    time.sleep(0.1)
    return response.choices[0].message.content

#def fetch_top_search_results(query, num_results=3):
#    top_results = search(query, num_results=num_results, advanced=True)
#    search_results = ""
#    for idx, result in enumerate(top_results, 1):
#        search_results += (f"{result.description}")
#    return search_results

def llm_call(system_instruct, contents, model_name="gemini-2.0-flash"):
        
    prompt = f"{system_instruct}\n\nProvided context: {contents}"
    model = gai.GenerativeModel(
    model_name=model_name,
    generation_config=gai.GenerationConfig(
    temperature=0.0,
    top_p=1.0,
    top_k=1,
    candidate_count=1,
    max_output_tokens=4096,
    )
    )
    response = model.generate_content(prompt)
    time.sleep(0.1)
    return response.text

#google_search_tool = Tool(
#    google_search = GoogleSearch()
#)

#def llm_call(system_instruct, contents, tool_list=[], temperature=0.0):
#    tools = []
#    if tool_list:
#        for t in tool_list:
#            if t == "web_search":
#                #tools.append({"web_search": {}})
#                tools.append(google_search_tool)
#                
#    client = genai.Client(api_key=GOOGLE_API_KEY)
#    response = client.models.generate_content(
#        model="gemini-2.0-flash",
#        config=types.GenerateContentConfig(
#            system_instruction=system_instruct,
#            tools=tools,
#            temperature=temperature,
#            ),
#        contents=contents
#    )
#    time.sleep(0.1)
#    return response.text

def query_certify_valid(user_query):
    system_instruction = dedent(f"""
                                Given the attached query below, decide whether the query can be answered with the available data. To decide on validity, remember that you are an agent which can answer questions about the Indian economy. 
                                
                                ## Valid topics and questions include: 
                                    Inflation (CPI), 
                                    wholesale prices (WPI), 
                                    industrial output (IIP), 
                                    manufacturing and other industrial sectors, 
                                    banking, finance, 
                                    GDP (gross domestic product), 
                                    state value added (GSVA), 
                                    state gross domestic product (GSDP),
                                    medium and small enterprises (MSME), 
                                    MSME queries around the world -- Asia, Europe, America,
                                    Credit related queries,
                                    agriculture and rural labour, 
                                    housing prices, 
                                    national income, 
                                    private income, 
                                    exports, imports, 
                                    macro and micro economic questions, 
                                    population or strength of labour force in various categories, and 
                                    Indian government initiatives regarding the economy. 
                                
                                If the query is very short without details about context (for example, "top 5 states"), then let it pass with a YES. 
                                
                                Queries should be marked as invalid only if they are clearly not related to the Indian economy.
                                
                                You MUST answer with a single word: YES (query is valid) or NO (query is invalid or out of bounds). Do NOT include any other thinking traces or text apart from YES or NO.
                                """)
    validity = llm_call(system_instruction, user_query).strip()
    return validity

def clarify_query(user_query):
    curdate = strftime("%Y-%m", gmtime())
    system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for an SQL RAG agent to pull the right data. 
                              
                    **** Rules to produce rephrased query ****
                        1. Do not edit the query as far as possible, only augment with a date range if none is present.
                            a. Remember that the current date is {curdate}.
                            b. If no date range is available from the context or from web search, use six months before {curdate} to {curdate}.
                            c. If the date range is LONGER than two years, then include the word "yearly" in the rephrase.
                            d. If the query asks about vague timelines such as "long term" without specifying dates, use the date two years ago from [{curdate}].
                            e. If the query asks for the impact of a specific event on certain quantities, use a date range of at most two years.
                            f. ALWAYS specify date ranges with both start date and end date.
                            g. NEVER change the dates if they are already mentioned in the query.
                            h. Always use month names and full years (e.g. April 2025, January 2022)
                            i. If the query date range is longer than 2 years (e.g. since 2022, in the last few years, in the last decade), mention "annual" in the rephrased query.
                            j. f the query talks about quarters (e.g. last two quarters), include "quarter" in the rephrased query.
                        2. If the query contains acronyms, include full form in parentheses.
                            Example: Query -> What is RBI's thought process in May 2025?
                                    Rephrased query -> What is RBI (Reserve Bank of India) thought process in May 2025?
                        4. Do not attach extraneous information apart from this, or include your own thinking traces. Keep it as close to the original query as possible.
                        5. IMPORTANT: Analyze the provided query for the existence of multiple entities, comparisons between quantities. For example, if the query asks about "contribution of Maharashtra to total GDP", rewrite it as "GDP of Maharashtra and GDP of India". If the query is about "apparel and leather", ensure that both "apparel" and "leather" are retained in the rephrased query.
                        6. If the word "India" is not mentioned in the query, include India in the rephrased query. 
                        7. If a statistic such as "top 5", "highest", "lowest", is mentioned in the query, this must be repeated in the rephrased query.
                        8. Some hints for rewrites:
                            - Queries related to "top states" should be mapped to GDP values, if no context is provided.
                            - Queries related to "top categories" should be mapped to inflation groups and subgroups, if no context is provided.
                            - Queries related to "top sectors" should be mapped to industrial output, if no context is provided.
                        11. You MUST restrict your output to at most 25 words.
                        12. VERY VERY IMPORTANT: Do not change the date range in {user_query} if it is already specified.
                        
                    **** Output format ****
                    Output as a strict json-like format, with the following entries
                        a. "Entity" such as inflation, GDP, etc.
                        b. "Category" such as food, mining, metals, current prices, etc.
                        c. "Qualifiers" such as list of states, top 5, etc.
                        d. "Frequency" from one of annual, quarterly, monthly
                        e. "Date range" with a minimum and maximum value, formatted in Month Year format such as "April 2024"
                        
                        Make sure keywords such as "states", "groups", "food", "labour", "agriculture", etc. are retained in the rephrased query. 
                        DO NOT miss out on any important words from the original query.
                        DO NOT include any thinking traces or text apart from the json format above.
            """),
    rephrased_query = llm_call_rephrase(system_instruction, user_query).strip()
    return rephrased_query

def old_clarify_query(user_query, table_list):
    curdate = strftime("%Y-%m", gmtime())
    system_instruction=dedent(f"""You are tasked with rephrasing the given query to make it easier for an SQL RAG agent to pull the right data. 
                              
                    Rules to produce rephrased query:
                        1. Do not edit the query as far as possible, only augment with a date range if none is present.
                            a. Remember that the current date is {curdate}.
                            b. If no date range is available from the context or from web search, use six months before {curdate} to {curdate}.
                            c. If the date range is LONGER than two years, then include the word "yearly" in the rephrase.
                            d. If the query asks about vague timelines such as "long term" without specifying dates, use the date five years ago from [{curdate}].
                            e. If the query contains an event without a date (for example, "COVID" or "51st meeting" or "the last world cup"), then use the google_search_tool to attach a date to the event.
                            f. If the query asks for the impact of a specific event on certain quantities, use a date range of at most two years.
                            g. ALWAYS specify date ranges with both start date and end date.
                            h. NEVER change the dates if they are already mentioned in the query.
                        2. If the query contains acronyms, include full form in parentheses.
                            Example: Query -> What is RBI's thought process in May 2025?
                                    Rephrased query -> What is RBI (Reserve Bank of India) thought process in May 2025?
                        3. Attach date range if no date range is available.
                            a. Include minimum and maximum date as far as possible (e.g. Feb 2025 to April 2025)
                            b. Always use month names and full years (e.g. April 2025, January 2022)
                            c. If the query date range is longer than 2 years (e.g. since 2022, in the last few years, in the last decade), mention "annual" in the rephrased query.
                            d. If the query talks about quarters (e.g. last two quarters), include "quarter" in the rephrased query.
                            e. If the query already contains a date range, do not edit the date range.
                        4. Do not attach extraneous information apart from this, or include your own thinking traces. Keep it as close to the original query as possible.
                        5. Some rules for contextualization:
                            a. We have data about inflation (CPI), gross domestic product (GDP), index of industrial production (IIP), Medium and Small Enterprises (MSME).
                            b. Within India, geographical divisions include All India and individual states such as Maharashtra, Karnataka, etc.
                            c. Data is typically available monthly, quarterly, annual (yearly).
                            d. If specific industries or categories of products are not mentioned, then include the word "General" in the rephrase.
                            e. If specific industries such as electrical, manufacturing, etc. or product groups such as food, clothing, vegetables, are mentioned, then include in the rephrase without changing.
                            f. If 
                        6. IMPORTANT: Analyze the provided query for the existence of multiple entities, comparisons between quantities. For example, if the query asks about "contribution of Maharashtra to total GDP", rewrite it as "GDP of Maharashtra and GDP of India". If the query is about "apparel and leather", ensure that both "apparel" and "leather" are retained in the rephrased query.
                        7. If the word "India" is not mentioned in the query, include India in the rephrased query. 
                        8. If a statistic such as "top 5", "highest", "lowest", is mentioned in the query, this must be repeated in the rephrased query.
                        9. ALWAYS attach date ranges to your query. If no dates are obvious, mention "latest available before {curdate}".
                        10. Some hints for rewrites:
                            - Queries related to "top states" should be mapped to GDP values, if no context is provided.
                            - Queries related to "top categories" should be mapped to inflation groups and subgroups, if no context is provided.
                            - Queries related to "top sectors" should be mapped to industrial output, if no context is provided.
                        11. You MUST restrict your output to at most 25 words.
                        12. VERY VERY IMPORTANT: Do not change the date range in {user_query} if it is already specified.
            """),
    rephrased_query = llm_call_rephrase(system_instruction, user_query).strip()#, tool_list=["web_search"]).strip()
    return rephrased_query

def generate_sql_queries(query):
    system_instruction=dedent("""Consider the query provided below. Your goal is to decorate the query and rewrite it in a form that can be used for searching through structured data. You must limit yourself to a maximum of 15 words for each query, and a STRICT MAXIMUM of 2 sub-queries.
            INSTRUCTIONS:
                1. Keep in mind that you are generating data retrieval requests related to the Indian economy. Focus on topics such as CPI (Consumer Price Inflation), GDP (Gross Domestic Product), IIP (Industrial Production), MSME (Medium and Small Enterprises) and others.
                2. Make sure you include all the key entities in at least one query.
                3. Make sure any commodities in the query are retained in your rephrased query.
                4. Unless specific date ranges are mentioned, keep the range of dates in each sub-query to APPROXIMATELY ONE YEAR AT MOST.
                5. If the impact of specific events is mentioned, generate subqueries for a few months before and after the event.
                6. If specific states in India are not mentioned, mention that the sub-query is for all India.
                7. If specific sub-categories of commodities are not mentioned, state that the sub-query is at broad category level.
                8. If the query duration is less than two years, generate separate sub-queries for annual and monthly data.
                9. DO NOT separate contiguous time periods for the same data into separate sub-queries.
                10. If you are asked for comparisons of categorical variables (top 4 states, top 5 categories, lowest 3 sectors), then FIX time-related fields (year, month, quarter, or similar) in the rephrased query.
                11. You MUST include time periods in your query. If no dates are obvious, mention "latest available before {curdate}".
                12. If the query asks about a specific date (e.g. "April 2024"), then DO NOT rephrase this date.
                
            EXAMPLE:
                Growth of (quantity) between 2020 and 2024 --> ["Growth of (quantity) from January 2020 to December 2024"]
                Growth of (quantity) [2019-06] to [2025-06] --> ["Growth of (quantity) from June 2019 to June 2025"]
                Correlation between IIP and GDP --> ["IIP in India in the last 3 years", "GDP in India in the last 3 years"]
                    
            **IMPORTANT:**
            - FORMATTING INSTRUCTIONS: Your output format should be a LIST OF STRINGS such as ["Sub-Query 1","Sub-Query 2"], with each string containing one sub-query. DO NOT output more than 2 sub-queries and stick to the word limit of 15 words per sub-query. DO NOT user more sub-queries than necessary, if 1 sub-query is sufficient. 
            - Do not split a single time period into multiple time periods. Separate sub-queries should only be used for different quantities and not for time periods!
            """)
    unitary_queries = llm_call(system_instruction, query)
    unitary_queries = unitary_queries.split("\"")[1:-1:2]
    return unitary_queries