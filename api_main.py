#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 18:06:45 2025

@author: harshad

Receive query to API for data retrieval and orchestrate all functions

To launch: 
    1. . ~/Projects/SR/venv/bin/activate
    2. nohup uvicorn api_main:app --host 0.0.0.0 --port 8001 --reload > debug.log 2>&1 &
"""
from   fastapi import FastAPI, HTTPException, Depends
from   fastapi.security.api_key import APIKeyHeader
from   pydantic import BaseModel
from   time import strftime, gmtime
import os
from   utils_common import clarify_query, generate_sql_queries, query_certify_valid
import time
import logging
from   datetime import datetime
from   handler_sql import batch_sql_queries, return_table_list
from   utils_common import llm_call
#from   async_handler_sql import batch_sql_queries

current_date = datetime.now().strftime('%Y-%m-%d')
API_KEY = os.getenv("ACQ_API_KEY")
API_KEY_NAME = "access_token"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)
app = FastAPI(title="Integrated retrieval server")

logging.basicConfig(
    filename = "orchestrate-"+current_date+".log",
    level=logging.INFO,  # Change to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

def has_long_run(s, min_run_length=200):
    """Detects long single-character runs (e.g., 'aaaaa...')"""
    if len(s) < min_run_length:
        return False
    count = 1
    for i in range(1, len(s)):
        count = count + 1 if s[i] == s[i-1] else 1
        if count >= min_run_length:
            return True
    return False

def compile_answer(user_query, responses):
    data_description = ""
    try:
        for i in range(len(responses)):
            if responses[i]["success"]:
                logger.info("<part> " + str(i+1) + " has data")
                data_description += "\n<part> " + str(i+1) + ":\n"
                data_description += "\nThe following data contains this information: "+ str(responses[i]["description"]) + "\nHere is the data:\n" + str(responses[i]["result"])
        if data_description != "":
            suggest_answer = llm_call(f"""Consider the following query: {user_query}. Given the data below, try to compile an answer to the query as best as you can. 
                                      1. Format the output as a markdown table where possible, and combine multiple parts marked by <part> tags into a single comprehensive answer. The table must be comprehensive and cover as much of the provided data as possible. Include every year, quarter, month present in the raw data.
                                      2. Do not add any of your reasoning traces, focus only on the answer to {user_query}. 
                                      3. Rules for answering the query:
                                          - If the data directly answers the query, simply repeat the answer. In this case, do not add any commentary, just print the data.
                                          - If the data is more extensive than what is needed to answer the query, then answer the query based on the provided data. DO NOT miss out on answering for the full date range. If the query is about years 2021 to 2024, all years from 2021, 2022, 2023, 2024 must be included in the summarized answer.
                                          - If there is insufficient data to answer the query {user_query} directly, mention this at the top and then answer the query as best as possible. Mention a reason for why the data might be insufficient.
                                      4. Retain the context of the different data points mentioned in the data, do not summarise unnecessarily.
                                      
                                      **DO NOT** carry out any mathematical operations or add/hallucinate your own information. Only provide a comprehensive answer by proper formatting of the response.
                                      """, data_description)
            return suggest_answer
        else:
            return "<insufficient_data>"
    except:
        return "Answer compilation failed"
    
def markdown_answer(user_query, responses, counter):
    if counter == 0:
        model_name = "gemini-2.0-flash"
    else:
        model_name = "gemini-2.5-flash-lite"
    data_description = ""
    try:
        for i in range(len(responses)):
            if responses[i]["success"]:
                logger.info("<part> " + str(i+1) + " has data")
                data_description += "\n<part> " + str(i+1) + ":\n"
                data_description += "\nThe following data contains this information description: "+ str(responses[i]["description"]) + "\nHere is the data:\n" + str(responses[i]["result"])
        if data_description != "":
            suggest_answer = llm_call(f"""Consider the following query: {user_query}. Given the data below, answer it as best as you can using a markdown format table.
                                      1. Give the table a "Caption" based on the information description provided. Any missing info can be briefly mentioned in the caption (in 6-10 words). Especially, for GDP by state, mention that not all states may have latest results.
                                      2. The table MUST be formatted in valid markdown format.
                                      3. Include all the information present in the data.
                                      4. Do not add any of your reasoning traces, focus only on the answer to {user_query}. 
                                      **DO NOT** carry out any mathematical operations or add/hallucinate your own information. Only provide a comprehensive answer by proper formatting of the response.
                                      """, data_description, model_name)
            return suggest_answer
        else:
            return "<insufficient_data>"
    except:
        return "<insufficient_data>"
    
def confidence_checker(user_query, suggested_answer):
    system_instruction = f"""
    Consider the following compiled answer: {suggested_answer}. It is meant to answer the user query attached below as provided context.
    
    Task: You must return a unique integer value from 0, 1, 2 based on the following criteria.
    0: If the compiled answer is completely irrelevant to the user query.
    1: If the compiled answer answers part of the user query, but not all of it.
    2: If the compiled answer fully answers the user query given below, and covers the time range requested.
    
    Remember to respond with a single digit from 0, 1, 2 without any other additional text.
    """
    confidence = llm_call(system_instruction, user_query).strip()
    try:
        confidence = str(int(confidence))
    except Exception as e:
        logger.warning("Could not infer confidence: " + str(confidence))
        logger.warning("Exception: " + str(e))
        confidence = "0"
    return confidence
    
async def verify_api_key(api_key: str = Depends(api_key_header)):
    print(f"Received API Key: {api_key}")
    print("Verify key: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Input model
class Question(BaseModel):
    question: str
    
class BatchRequest(BaseModel):
    queries: list[str]

@app.post("/integrated_query", dependencies=[Depends(verify_api_key)])
async def orchestrate(question: Question):
    try:
        start_time = time.time()
        user_query = str(question.question).strip()
        logger.info("Original received query: " + str(user_query))
        #all_tables = return_table_list()
        validity = query_certify_valid(user_query)
        if "NO" in validity:
            total_time = time.time() - start_time
            logger.info(f"Total processing time: {total_time}")
            logger.info("Query is out of bounds")
            result = {"success": False}
            return {"query": user_query, "suggested_answer": "Out of bounds", "context": "N/A", "urls": [], "references": [], "total_time": total_time,
                    "response": result, "confidence": "0"}
        rephrased_query = clarify_query(user_query)
        logger.info("Rephrased query:\n" + rephrased_query)
        sql_queries = generate_sql_queries(rephrased_query)
        batch = BatchRequest(queries=sql_queries)
        sql_responses = await batch_sql_queries(batch, user_query)
        has_garbage = True
        counter = 0
        while has_garbage and (counter < 2):    
            suggest_answer = markdown_answer(user_query, sql_responses["responses"], counter)
            has_garbage = has_long_run(str(suggest_answer))
            if has_garbage:
                logger.warning("Output contained garbage, retrying")
            counter += 1
        if has_garbage:
            total_time = time.time() - start_time
            result = {"success": False}
            return {"query": user_query, "suggested_answer": "Model throwing garbage tokens", "context": "N/A", "urls": [], "references": [], "total_time": total_time,
                    "response": result, "confidence": "0"}
        #suggest_answer = compile_answer(user_query, sql_responses["responses"])
        commentary = "Table information:\n"
        logger.info("Suggested answer: ")
        logger.info(suggest_answer)
        logger.info("Commentary:")
        logger.info(commentary)
        confidence = confidence_checker(user_query, suggest_answer)
        logger.info("Confidence: " + str(confidence))
        urls = []
        refs = []
        meta = []
        try:
            urls = [
                resp.get("url")
                for resp in sql_responses["responses"]
                if resp.get("success") is True and "url" in resp
            ]
            #urls = list(set(urls))
            refs = [
                resp.get("reference")
                for resp in sql_responses["responses"]
                if resp.get("success") is True and "reference" in resp
            ]
            #urls = list(set(urls))
            meta = [
                resp.get("table_metadata")
                for resp in sql_responses["responses"]
                if resp.get("success") is True and "table_metadata" in resp
            ]
            commentary += str(meta)
        except:
            logger.info("Could not trace back urls or references")
        
        total_time = time.time() - start_time
        logger.info(f"Total processing time: {total_time}")
        return {"query": user_query, "rephrased_query": rephrased_query, "suggested_answer": suggest_answer, "context": commentary, "urls": urls, "references": refs, "confidence": str(confidence), "sql_queries": sql_queries, "total_time": total_time,
                "response": sql_responses}
    except Exception as error:
        logger.error(f"Error: {error}")
        total_time = time.time() - start_time
        return {"query": user_query, "rephrased_query": rephrased_query, "suggested_answer": "Answer compilation failed", "context": "N/A", "urls": [], "references": [], "confidence": "0", "sql_queries": sql_queries, "total_time": total_time,
                "response": []}