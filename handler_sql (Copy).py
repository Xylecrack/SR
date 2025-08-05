import logging
from   datetime import datetime
from   time import strftime, gmtime
from   threading import Lock
from   langchain_community.utilities import SQLDatabase
from   sqlalchemy.pool import QueuePool
import re
import json
import time
from   pydantic import BaseModel
import asyncio
from   fastapi import HTTPException
from   utils_sql import classify_query, file_selector_CPI, file_selector_GDP, file_selector_IIP, file_selector_MSME, generate_sql_query, handle_pandas_response, table_citation, identify_generic_columns
from   sqlalchemy import create_engine, text
import pandas as pd
import ast
from   utils_common import llm_call, llm_call_rephrase

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

DATABASE_URI = "postgresql://postgres:admin@localhost:5432/final"
db = SQLDatabase.from_uri(
    DATABASE_URI,
    engine_args={
        "poolclass": QueuePool,
        "pool_size": 20,
        "max_overflow": 10,
        "pool_recycle": 3600,
        "pool_timeout": 30
    }
)

def split_conditions(where_clause):
    # Split on 'AND' or 'OR' that is not within quotes
    pattern = r'\b(?:AND|OR)\b(?=(?:[^\'"]*[\'"][^\'"]*[\'"])*[^\'"]*$)'
    conditions = re.split(pattern, where_clause, flags=re.IGNORECASE)
    return [cond.strip() for cond in conditions if cond.strip()]

def extract_conditions(where_clause):
    # Handle BETWEEN ... AND ... expressions
    between_pattern = r"BETWEEN\s+('[^']*'|\d+(\.\d+)?|\w+)\s+AND\s+('[^']*'|\d+(\.\d+)?|\w+)"
    between_matches = re.findall(between_pattern, where_clause)
    placeholders = {}
    for i, match in enumerate(between_matches):
        full_match = f"BETWEEN {match[0]} AND {match[2]}"
        placeholder = f"__BETWEEN_{i}__"
        placeholders[placeholder] = full_match
        where_clause = where_clause.replace(full_match, placeholder)

    # Split by AND/OR only if not inside quotes or parentheses
    conditions = []
    current = ''
    stack = []
    in_single_quote = False
    in_double_quote = False
    i = 0
    while i < len(where_clause):
        c = where_clause[i]
        if c == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            current += c
        elif c == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            current += c
        elif c == '(' and not in_single_quote and not in_double_quote:
            stack.append('(')
            current += c
        elif c == ')' and not in_single_quote and not in_double_quote:
            if stack:
                stack.pop()
            current += c
        elif (where_clause[i:i+4].upper() == ' AND' and not stack and not in_single_quote and not in_double_quote):
            conditions.append(current.strip())
            current = ''
            i += 3
        elif (where_clause[i:i+3].upper() == ' OR' and not stack and not in_single_quote and not in_double_quote):
            conditions.append(current.strip())
            current = ''
            i += 2
        else:
            current += c
        i += 1
    if current.strip():
        conditions.append(current.strip())

    # Restore BETWEEN ... AND ... expressions
    for i, part in enumerate(conditions):
        for placeholder, expr in placeholders.items():
            if placeholder in part:
                conditions[i] = part.replace(placeholder, expr)
    return conditions

def clean_sql_query(raw_sql: str, table_name: str, engine, logger, error="N/A") -> str:
    if ";" in raw_sql:
        raw_sql = raw_sql.replace(";","")
    
    if "where" not in raw_sql.lower():
        return raw_sql
    
    idx = raw_sql.upper().find('LIMIT')
    if idx != -1:
        raw_sql = raw_sql[:idx].rstrip()  # Remove LIMIT and anything after

    # Split at WHERE
    pre_where, where_and_after = re.split(r'\bwhere\b', raw_sql, flags=re.IGNORECASE, maxsplit=1)

    # Split at ORDER BY if present
    where_clause, *order_by_clause = re.split(r'\border by\b', where_and_after, flags=re.IGNORECASE, maxsplit=1)

    # Now split conditions in where_clause
    conditions = extract_conditions(where_clause)
    
    valid_conditions = []
    with engine.connect() as conn:
        for condition in conditions:
            if (("month" in condition) or ("quarter" in condition)):
                logger.info("Dropping quarter-based or month-based condition: " + condition)
                continue
            if (('data_updated_date' in condition) or ('data_release_date' in condition) or ('data_source' in condition)):
                logger.info("Dropping metadata based condition: " + condition)
                continue
            if (error != "N/A") and ("year" not in condition):
                logger.info("Dropping non time-based condition since running after error")
                continue
            try:
                validation_query = text(f"SELECT 1 FROM {table_name} WHERE " + condition + " LIMIT 1")
                result = conn.execute(validation_query)
                if result.fetchone():
                    valid_conditions.append(condition)
                else:
                    logger.info(f"Dropping invalid condition: {condition}")
            except Exception as e:
                logger.warning(f"Error validating condition '{condition}': {e}")
                #valid_conditions.append(condition)
                
    # Reconstruct the query
    result_query = f"{pre_where.strip()} WHERE {' AND '.join(valid_conditions)}"
    if order_by_clause:
        result_query += f" ORDER BY {order_by_clause[0].strip()}"
    return result_query

def old_clean_sql_query(raw_sql: str, table_name: str, engine, logger) -> str:
    if ";" in raw_sql:
        raw_sql = raw_sql.replace(";","")
    
    if "where" not in raw_sql.lower():
        return raw_sql
    
    idx = raw_sql.upper().find('LIMIT')
    if idx != -1:
        raw_sql = raw_sql[:idx].rstrip()  # Remove LIMIT and anything after

    # Split at WHERE
    pre_where, where_and_after = re.split(r'\bwhere\b', raw_sql, flags=re.IGNORECASE, maxsplit=1)

    # Split at ORDER BY if present
    where_clause, *order_by_clause = re.split(r'\border by\b', where_and_after, flags=re.IGNORECASE, maxsplit=1)

    # Now split conditions in where_clause
    conditions = split_conditions(where_clause)

    valid_conditions = []
    with engine.connect() as conn:
        for condition in conditions:
            match = re.match(r"(\w+)\s*=\s*['\"]?(.+?)['\"]?$", condition)
            if not match:
                logger.info(f"Skipping unparsable condition: {condition}")
                if (("month" in condition) or ("quarter" in condition)):
                    logger.info("Dropping quarter-based or month-based condition")
                else:
                    if (('data_updated_date' in condition) or ('data_release_date' in condition) or ('data_source' in condition)):
                        logger.info("Dropping metadata based condition")
                    else:
                        valid_conditions.append(condition)
                continue

            column, value = match.groups()
            try:
                validation_query = text(f"SELECT 1 FROM {table_name} WHERE {column} = :val LIMIT 1")
                result = conn.execute(validation_query, {"val": value})
                if result.fetchone():
                    valid_conditions.append(condition)
                else:
                    logger.info(f"Dropping invalid condition: {condition}")
            except Exception as e:
                logger.warning(f"Error validating condition '{condition}': {e}")
                valid_conditions.append(condition)

    # Reconstruct the query
    result_query = f"{pre_where.strip()} WHERE {' AND '.join(valid_conditions)}"
    if order_by_clause:
        result_query += f" ORDER BY {order_by_clause[0].strip()}"
    return result_query

def handle_json_response(raw_data):
    try:
        if not isinstance(raw_data, str):
            raise Exception("Raw data must be a string.")
        if "summarized_info" in raw_data:
            py_dict = ast.literal_eval(raw_data)
            return json.loads(json.dumps(py_dict,default=str))
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
        try:
            return json.loads(json.dumps(raw_data,default=str))
        except:
            raise Exception(f"Error processing JSON: {str(e)}")
        
def return_table_list():
    try:
        engine = create_engine(DATABASE_URI)
        connection = engine.connect()
        table_list_query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
              AND (table_type='BASE TABLE' OR table_type='VIEW');
        """
        table_list_return = connection.execute(text(table_list_query))
        # Fetch all the results
        table_list = table_list_return.fetchall()
        table_list_return = str([table[0] for table in table_list])
        #logger.info("Found tables: " + table_list_return)
        return table_list_return
    except:
        logger.info("Could not fetch tables")
        return ""
    
async def process_single_query(unit_query: str, orig_query: str):

    """Convert user query to SQL and execute it using an agent with table context."""
    with counter_lock:
        query_id = query_counter["value"]
        query_counter["value"] += 1
    curdate = strftime("%Y-%m", gmtime())

    table_info  = "N/A"
    start_time = time.time()
    logger.info(f"START: Processing query:query_id:{query_id}: {unit_query}")    
    logger.info("Start processing: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))
    unit_query = unit_query.strip()
    logger.info(f"Received unitary query: {unit_query}")
    
    query_class = classify_query(unit_query)
    logger.info(f"Query class: {query_class}")
    
    if query_class == "CPI":
        selected_file = file_selector_CPI(unit_query).strip()
        ref_url = "https://esankhyiki.mospi.gov.in/macroindicators?product=cpi"
        logger.info(f"Selected file: {selected_file}")
        
    if query_class == "GDP":
        selected_file = file_selector_GDP(unit_query).strip()
        ref_url = "https://esankhyiki.mospi.gov.in/macroindicators?product=nas"
        logger.info(f"Selected file: {selected_file}")
        
    if query_class == "IIP":
        selected_file = file_selector_IIP(unit_query).strip()
        ref_url = "https://esankhyiki.mospi.gov.in/macroindicators?product=iip"
        logger.info(f"Selected file: {selected_file}")
        
    if query_class == "MSME":
        selected_file = file_selector_MSME(unit_query).strip()
        ref_url = "https://msme.gov.in/"
        logger.info(f"Selected file: {selected_file}")
        
    if (query_class != "CPI") and (query_class != "GDP") and (query_class != "IIP") and (query_class != "MSME"):
        selected_file = "none_of_these"
        ref_url = "N/A"

    if selected_file.strip() == "none_of_these":
        total_time = time.time() - start_time
        ref_name = "N/A"
        return {
                "unit_query": unit_query,
                "result": [],
                "description": "No data",
                "success": False,
                "message": "We do not have structured data related to this query",
                "unitary_time": total_time,
                "reference": "N/A",
                "url": "N/A",
                "table_metadata": table_info,
            }
    
    try:
        ref_name = table_citation(selected_file)
        engine = create_engine(DATABASE_URI)
        # Query the information schema to get the table schema
        schema_query = f"""
        SELECT column_name, data_type, is_nullable, column_default
        FROM information_schema.columns
        WHERE table_name = '{selected_file}'
        ORDER BY ordinal_position;
        """
        logger.info("Generated schema query:")
        logger.info(schema_query)
        
        max_retries = 3
        attempt     = 1
        success     = False
        error       = "N/A"
        max_rows    = 500
        
        while (not success) and (attempt <= max_retries):
            result=None
            error = "N/A"
            try:
                logger.info(f"Attempt {attempt}:query_id:{query_id}: to process query")
                # Connect to the database
                connection = engine.connect()
                context = ""
                # Execute the query
                schema_result = connection.execute(text(schema_query))
                # Fetch all the results
                schema = schema_result.fetchall()
                context += "Schema:\n"
                # Print the schema
                for column in schema:
                    if """'id'""" not in str(column):
                        context += str(column) + "\n"
                schema = str(schema)
                contains_year = ""
                if "'date_stamp'" in schema:
                    contains_year = 'date_stamp'
                elif "'year'" in schema:
                    contains_year = 'year'
                elif "'years'" in schema:
                    contains_year = 'years'
                contains_month = ""
                if "'month_numeric'" in schema:
                    contains_month = 'month_numeric'
                # Get generic columns and their suggested values
                try:
                    gen_cols = identify_generic_columns(schema)
                    if gen_cols:
                        gen_col_string = ""
                        for g in range(len(gen_cols)):
                            gen_col_string += str(gen_cols[g])
                            if g < len(gen_cols)-1:
                                gen_col_string += ", "
                        pull_sample_query = "SELECT " + gen_col_string + " FROM " + selected_file + " ORDER BY RANDOM() LIMIT 40;"
                        logger.info("Trying to run: " + pull_sample_query)
                        sample_result = connection.execute(text(pull_sample_query))
                        sample_cat = ""
                        for row in sample_result:
                            sample_cat += str(row) + "\n"
                        #print("Got result \n" + sample_cat)
                        value_list = "\n\nDistinct value lists:"
                        for col in gen_cols:
                            distinct_vals = []
                            #set_vals = ""
                            sample_result = connection.execute(
                                text(f"SELECT DISTINCT {col} FROM {selected_file}")
                            )
                            for row in sample_result:
                                distinct_vals.append(str(row))
                            distinct_vals = [ast.literal_eval(t)[0] for t in distinct_vals]
                            #print("\nValues for " + col + ": " + str(distinct_vals))
                            value_list += f"\nValues for {col}: {distinct_vals}"
                        set_vals = llm_call_rephrase(f""""
                                            Consider the following sample rows for columns: {gen_col_string}.
                                            Your task is to return a set of assignments for each columns which can help minimize the number of rows pulled by an SQL agent.
                                            {sample_cat}
                                            Return the set values based on the following user query. 
                                            ## Rule:
                                            - Look for values such as General, Combined, * where the query below does not specify anything. 
                                            - Do not include any other text in your response, apart from suggested assignments.
                                            - REMOVE any conditions based on release_date or updated_date.
                                            - Do not use any settings apart from the unique lists given below.
                                            - Do not use any state, group, category names not mentioned in the query below.
                                            
                                            - IMPORTANT NEVER choose more than 5 values for any single filter
                                            - IMPORTANT If the query mentions "top states", "Indian states", "GDP", DO NOT filter on the state column!
                                            - IMPORTANT If the query mentions an entity such as "states", "inflation", "category", "group" WITHOUT specifying a value for this entity, then DO NOT include a filter based on this entity.
                                            
                                            Lists of unique values are as follows:\n{value_list}.\nThe query to be handled is:""", unit_query)
                        logger.info("Suggested value sets: " + str(set_vals))
                        context += "\n\nSuggested value settings:\n" + str(set_vals)
                except:
                    logger.warning("Failed to process suggested value settings")
                # Close the connection
                connection.close()
                n = 40 #50 + 25*(attempt-1)  # Replace with your desired number of rows
                context += "\n\nSample rows:\n"
                # Run the query
                with engine.connect() as connection:
                    sample_result = connection.execute(
                        text(f"SELECT * FROM {selected_file} ORDER BY RANDOM() LIMIT :n"),
                        {"n": n}
                    )
                    # Fetch and print the rows
                    for row in sample_result:
                        context += str(row) + "\n" #print(row)
                logger.info(context)
                if error == "N/A":
                    response, query_for_table = generate_sql_query(unit_query, schema, context, selected_file)
                else:
                    #unit_query = retry_query(unit_query, error)
                    #logger.info(f"Changed user query to: {unit_query}")
                    response, query_for_table = generate_sql_query(unit_query, schema, context, selected_file, error)
                logger.info(f"Used query for specific table: {query_for_table}")
                logger.info("SQL query:")
                try:
                    response = response.split("```")[1]
                    response = response.split("sql")[1]
                    response = response.split(";")[0]
                except:
                    logger.warning("Could not remove decorator from original query")
                if "\n" in response:
                    response = response.replace("\n"," ")
                logger.info("Originally: " + response)
                try:
                    #response = llm_call_rephrase(f"""Consider the following query: {orig_query}.\nThe following SQL query is intended to retrieve relevant data pertaining to it. Here are some column setting conditions: {set_vals}. Remember that financial year involves previous and current year (e.g. FY25 = 2024 and 2025). Based ONLY on a consideration of the date range (keep in mind current date is {curdate}), provide a revised SQL query. Make edits only if needed, and change only the date range. Return your response ONLY as a valid SQL query, with NO DECORATIVE TEXT.
                                                 
                    #REMEMBER THAT YOU SHOULD ONLY EDIT THE DATE/YEAR BASED CONDITIONS USING THE SCHEMA {schema}""", response)
                    response = llm_call(f"""Consider the following query: {orig_query}.
                                        
                    The following SQL query is intended to retrieve relevant data pertaining to it. 
                                                
                    REMEMBER THAT YOU SHOULD ONLY EDIT THE DATE/YEAR BASED CONDITIONS USING THE SCHEMA {schema}

                    Here are some column setting conditions: {set_vals}. Remember that financial year involves previous and current year (e.g. FY25 = 2024 and 2025). 

                    Based ONLY on a consideration of the date range (keep in mind current date is {curdate}), provide a revised SQL query. 

                    Make edits only if needed, and change only the date range. 
                                                 
                    ** IMPORTANT RULE ** Make sure you REMOVE ANY MONTH BASED (month = or month_numeric =) conditions.

                    ** IMPORTANT RULE ** Return your response ONLY as a valid SQL query, with NO DECORATIVE TEXT.""", response)
                except:
                    #response = llm_call_rephrase(f"""Consider the following query: {orig_query}.\nThe following SQL query is intended to retrieve relevant data pertaining to it. Remember that financial year involves previous and current year (e.g. FY25 = 2024 and 2025). Based ONLY on a consideration of the date range (keep in mind current date is {curdate}), provide a revised SQL query. Make edits only if needed, and change only the date range. Return your response ONLY as a valid SQL query, with NO DECORATIVE TEXT.
                                                 
                    #REMEMBER THAT YOU SHOULD ONLY EDIT THE DATE/YEAR BASED CONDITIONS USING THE SCHEMA {schema}""", response)
                    response = llm_call(f"""Consider the following query: {orig_query}.
                                        
                    The following SQL query is intended to retrieve relevant data pertaining to it. 
                                                
                    REMEMBER THAT YOU SHOULD ONLY EDIT THE DATE/YEAR BASED CONDITIONS USING THE SCHEMA {schema}

                    Based ONLY on a consideration of the date range (keep in mind current date is {curdate}), provide a revised SQL query. 

                    Make edits only if needed, and change only the date range. 
                                                 
                    ** IMPORTANT RULE ** Make sure you REMOVE ANY MONTH BASED (month = or month_numeric =) conditions.

                    ** IMPORTANT RULE ** Return your response ONLY as a valid SQL query, with NO DECORATIVE TEXT.""", response)
                try:
                    response = response.split("```")[1]
                    response = response.split("sql")[1]
                    response = response.split(";")[0]
                except:
                    logger.warning("Could not remove decorator from validated query")
                if "\n" in response:
                    response = response.replace("\n"," ")
                #if "```" in response:
                #    response = response.split("```")[1]
                logger.info("After validation: " + str(response))
                try:
                    response = clean_sql_query(response, selected_file, engine, logger, error)
                    logger.info("Cleaned SQL: " + response)
                except Exception as e:
                    logger.warning(f"Query cleaning failed. Continuing with original SQL. Error: {e}")
                if "ORDER" not in str(response):
                    if contains_year != "":
                        if contains_month != "":
                            response = str(response) + f"\nORDER BY {contains_year} DESC, {contains_month} DESC"
                        else:
                            response = str(response) + f"\nORDER BY {contains_year} DESC"
                if "LIMIT" not in str(response):
                    response = str(response) + f"\nLIMIT {max_rows};"
                else:
                    response = str(response)
                    response = re.sub(r'\blimit\s+\d+\b', 'LIMIT 100', response, flags=re.IGNORECASE)
                response = "SELECT * \nFROM" + response.split("FROM",1)[1]
                if "`" in response:
                    response = response.replace("`", "'")
                #partq = response.split("FROM")[0]
                #if " AS " in partq: 
                #    error = "Trying to set AS in SELECT condition is not allowed"
                #    raise Exception(error)
                logger.info(response)
                logger.info("SQL response:")
                
                with engine.connect() as connection:
                    df = pd.read_sql(text(response), connection)
                    nrows = len(df)
                    logger.info("Number of rows pulled: " + str(nrows))
                            
                if nrows == 0:
                    error = "SQL query resulted in no data. Try changing categories or broadening time scope, or REDUCING the number of filters: " + str(response)
                    logger.info(error)
                else:
                    error = "N/A"
                    df.fillna('', inplace=True) 
                if error == "N/A":
                    try:
                        reference_query = f"""
                                SELECT source, source_url, business_metadata FROM tables_metadata WHERE table_name = '{selected_file}';
                                """
                        with engine.connect() as connection:
                            ref_result = connection.execute(text(reference_query))
                            references = ref_result.fetchall()
                            ref_name, ref_url, table_info = references[0]
                            table_info = str(table_info)
                            logger.info("Got metadata: " + ref_name + ", " + ref_url + ", " + table_info)
                    except Exception as e:
                        logger.warning("Could not fetch table metadata")
                        logger.warning(str(e))
                    if 'data_source' in df.columns:
                        ref_name = df.loc[0, 'data_source']
                        logger.info("Found a data source column: " + ref_name)
                        df = df.drop(columns=['data_source'])
                    result, headers = handle_pandas_response(df, unit_query, orig_query, max_rows)
                    success = True
                else:
                    attempt += 1
            except Exception as error:
                logger.info("Something went wrong in SQL retrieval")
                error = str(error) + str(response)
                logger.info(error)
                result = []
                attempt += 1
        logger.info("Got result: Current time: " + strftime("%Y-%m-%d %H-%M-%S", gmtime()))

        if not result: # or not isinstance(result, str):
            raise Exception("No output found from agents.")

        logger.info(str(result))
        logger.info(str(headers))
        parsed_data = handle_json_response(str(result))
        
        if not parsed_data:
            raise Exception("Parsed data is None or empty.")

        # Successful execution
        logger.info(parsed_data)
        logger.info(headers)
        total_time = time.time() - start_time

        logger.info(f"Response:query_id:{query_id}: {parsed_data}")
        logger.info(f"Total processing time:query_id:{query_id}: {total_time:.2f} seconds")

        if isinstance(parsed_data, list) and parsed_data and isinstance(parsed_data[0], dict):
            if any(key in parsed_data[0] for key in ["message", "status", "error"]):
                total_time = time.time() - start_time
                return {
                        "unit_query": unit_query,
                        "result": [],
                        "description": "No data",
                        "success": False,
                        "message": "Query resulted in error",
                        "unitary_time": total_time,
                        "reference": "N/A",
                        "url": "N/A",
                        "table_metadata": table_info,
                    }
            
        return {
                "unit_query": unit_query,
                "result": [parsed_data],
                "description": headers,
                "success": True,
                "message": "Successful completion",
                "unitary_time": total_time,
                "reference": ref_name,
                "url": ref_url,
                "table_metadata": table_info,
            }
        
    except HTTPException as e:
        raise e
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
                "unitary_time": total_time,
                "reference": "N/A",
                "url": "N/A",
                "table_metadata": table_info,
            }
    
    total_time = time.time() - start_time
    return {
            "unit_query": unit_query,
            "result": [],
            "description": "No data",
            "success": False,
            "message": "Reached end of function",
            "unitary_time": total_time,
            "reference": "N/A",
            "url": "N/A",
            "table_metadata": table_info,
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