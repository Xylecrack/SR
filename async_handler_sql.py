#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 31 16:36:56 2025

@author: harshad
"""

from sqlalchemy.ext.asyncio import create_async_engine #, AsyncEngine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import logging
from   datetime import datetime
from   time import strftime, gmtime
from   threading import Lock
#from   langchain_community.utilities import SQLDatabase
#from   sqlalchemy.pool import QueuePool
import re
import json
import time
from   pydantic import BaseModel
import asyncio
from   fastapi import HTTPException
from   utils_sql import classify_query, file_selector_CPI, file_selector_GDP, file_selector_IIP, file_selector_MSME, generate_sql_query, retry_query, handle_pandas_response
from   sqlalchemy import create_engine, text
import pandas as pd
import ast

DATABASE_URL_SYNC = "postgresql://postgres:admin@localhost:5432/final"
sync_engine = create_engine(DATABASE_URL_SYNC)

async def read_sql_in_thread(sql: str):
    return await asyncio.to_thread(pd.read_sql_query, sql, sync_engine)

DATABASE_URL_ASYNC = "postgresql+asyncpg://postgres:admin@localhost:5432/final"

async_engine = create_async_engine(
    DATABASE_URL_ASYNC,
    pool_size=20,
    max_overflow=10,
    pool_recycle=3600,
    pool_timeout=30,
)

async_session = sessionmaker(bind=async_engine, class_=AsyncSession, expire_on_commit=False)

query_counter = {"value": 1}
counter_lock  = Lock()
current_date  = datetime.now().strftime('%Y-%m-%d')
QUERY_TIMEOUT = 60  # seconds
 
logging.basicConfig(
    filename = "sql-"+current_date+".log",
    level=logging.INFO,  # Change to DEBUG for more details
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

#DATABASE_URI = "postgresql://postgres:admin@localhost:5432/final"
#db = SQLDatabase.from_uri(
#    DATABASE_URI,
#    engine_args={
#        "poolclass": QueuePool,
#        "pool_size": 20,
#        "max_overflow": 10,
#        "pool_recycle": 3600,
#        "pool_timeout": 30
#    }
#)

def handle_json_response(raw_data):
    try:
        if not isinstance(raw_data, str):
            raise Exception("Raw data must be a string.")
        if "summarized_info" in raw_data:
            py_dict = ast.literal_eval(raw_data)
            return json.loads(json.dumps(py_dict))
        logger.info("Handle JSON: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))
        # Remove code fences if present
        raw_data = raw_data.replace("```json", "").replace("```", "").strip()

        # Fix common formatting issues
        raw_data = re.sub(r"'", '"', raw_data)  # Single to double quotes
        raw_data = re.sub(r",\s*([}\]])", r"\1", raw_data)  # Remove trailing commas

        # Try to extract multiple JSON objects (best-effort)
        object_matches = re.findall(r'\{[^{}]*\}', raw_data, re.DOTALL)

        valid_objects = []
        for obj_str in object_matches:
            try:
                obj = json.loads(obj_str)
                valid_objects.append(obj)
            except json.JSONDecodeError:
                continue  # skip broken ones

        if not valid_objects:
            raise Exception("No valid JSON objects found.")

        return valid_objects

    except Exception as e:
        raise Exception(f"Error processing JSON: {str(e)}")

async def process_single_query(unit_query: str, orig_query: str):
    with counter_lock:
        query_id = query_counter["value"]
        query_counter["value"] += 1

    start_time = time.time()
    logger.info(f"START: Processing query:query_id:{query_id}: {unit_query}")    
    logger.info("Start processing: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))
    unit_query = unit_query.strip()
    logger.info(f"Received unitary query: {unit_query}")
    
    query_class = classify_query(unit_query)
    logger.info(f"Query class: {query_class}")

    # File selector
    if query_class == "CPI":
        selected_file = file_selector_CPI(unit_query).strip()
    elif query_class == "GDP":
        selected_file = file_selector_GDP(unit_query).strip()
    elif query_class == "IIP":
        selected_file = file_selector_IIP(unit_query).strip()
    elif query_class == "MSME":
        selected_file = file_selector_MSME(unit_query).strip()
    else:
        selected_file = "none_of_these"

    logger.info(f"Selected file: {selected_file}")

    if selected_file == "none_of_these":
        total_time = time.time() - start_time
        return {
            "unit_query": unit_query,
            "result": [],
            "description": "No data",
            "success": False,
            "message": "We do not have structured data related to this query",
            "unitary_time": total_time
        }

    try:
        schema_query = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = '{selected_file}'
        ORDER BY ordinal_position;
        """

        max_retries = 3
        attempt = 1
        error = "N/A"
        success = False
        max_rows = 500

        while not success and attempt <= max_retries:
            result = None
            try:
                logger.info(f"Attempt {attempt}:query_id:{query_id}: to process query")
                context = ""

                # Fetch schema
                async with async_engine.connect() as conn:
                    schema_result = await conn.execute(text(schema_query))
                    schema = schema_result.fetchall()

                context += "Schema:\n" + "\n".join(str(col) for col in schema)

                # Sample rows
                sample_query = f"SELECT * FROM {selected_file} ORDER BY RANDOM() LIMIT :n"
                sample_rows = []
                async with async_engine.connect() as conn:
                    result = await conn.execute(text(sample_query), {"n": 50 + 25 * (attempt - 1)})
                    sample_rows = result.fetchall()

                context += "\n\nSample rows:\n" + "\n".join(str(row) for row in sample_rows)
                logger.info(context)

                # LangChain SQL generation
                if error == "N/A":
                    response, query_for_table = generate_sql_query(unit_query, str(schema), context, selected_file)
                else:
                    unit_query = retry_query(unit_query, error)
                    logger.info(f"Retrying with revised query: {unit_query}")
                    response, query_for_table = generate_sql_query(unit_query, str(schema), context, selected_file, error)

                logger.info(f"Used query for specific table: {query_for_table}")
                response = response.split("```")[1].split("sql")[1].split(";")[0]
                if "LIMIT" not in response.upper():
                    response += f"\nLIMIT {max_rows};"
                response = "SELECT * \nFROM" + response.split("FROM")[1]
                logger.info(f"Generated SQL query:\n{response}")

                # Execute query via sync connection using asyncio.to_thread
                df = await read_sql_in_thread(response)
                nrows = len(df)
                logger.info(f"Number of rows pulled: {nrows}")

                if nrows == 0:
                    error = "SQL query resulted in no data. Try changing categories or broadening time scope, or reducing filters."
                else:
                    result, headers = handle_pandas_response(df, unit_query, orig_query, max_rows)
                    success = True

            except Exception as e:
                logger.exception("Exception during SQL query execution.")
                error = str(e)
                result = []
                attempt += 1

        if not result:
            raise Exception("No output found from agents.")

        parsed_data = handle_json_response(str(result))

        if not parsed_data:
            raise Exception("Parsed data is None or empty.")

        total_time = time.time() - start_time
        logger.info(f"Response:query_id:{query_id}: {parsed_data}")
        logger.info(f"Total processing time: {total_time:.2f} seconds")

        return {
            "unit_query": unit_query,
            "result": [parsed_data],
            "description": headers,
            "success": True,
            "message": "Successful completion",
            "unitary_time": total_time
        }

    except Exception as e:
        error_message = str(e)
        logger.error(f"Error: {error_message}")
        total_time = time.time() - start_time
        return {
            "unit_query": unit_query,
            "result": [],
            "description": "No data",
            "success": False,
            "message": error_message,
            "unitary_time": total_time
        }


class BatchRequest(BaseModel):
    queries: list[str]
    
semaphore = asyncio.Semaphore(5)

async def batch_sql_queries(batch: BatchRequest, orig_query: str):
    if len(batch.queries) > 5:
        raise HTTPException(status_code=400, detail="Maximum of 5 queries allowed per batch.")

    # override per-batch semaphore
    
    async def limited(q, orig_query):
        async with semaphore:
            try:
                return await asyncio.wait_for(process_single_query(q, orig_query), timeout=QUERY_TIMEOUT)
            except asyncio.TimeoutError:
                logger.info("Query timed out: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))
                return {
                    "error": "Query timed out.",
                    "unit_query": q,
                    "success": False,
                }               

    tasks = [limited(q, orig_query) for q in batch.queries]
    results = await asyncio.gather(*tasks , return_exceptions=True)

    successes = sum(1 for r in results if isinstance(r, dict) and r.get('success'))
    return {
        "processed": len(results),
        "successful": successes,
        "errors": len(results) - successes,
        "responses": results
    }