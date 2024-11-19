import json
import os
import pandas as pd
import time

pd.set_option('mode.chained_assignment', None)
from base import BASE_DIR

def load_data():
    """
    Load base data from CUAD.json
    """
    with open(f'{BASE_DIR}/input/CUAD.json', 'r') as f:
        data = json.load(f)["data"]
        return data

def safe_load_json(response_text, default_value=None):
    """
    Safely loads JSON data from a response text, trimming extra data outside JSON if necessary.

    Args:
        response_text (str): The JSON string to be loaded.
        default_value (any): Value to return in case of an error (default: None).

    Returns:
        dict: The loaded JSON data if successful, otherwise default_value.
    """
    try:
        # Find the first and last curly brackets and extract JSON content
        start = response_text.find("{")
        end = response_text.rfind("}") + 1
        
        # If valid JSON boundaries are found, trim the response text to only JSON
        if start != -1 and end != -1:
            trimmed_text = response_text[start:end]
            json_data = json.loads(trimmed_text)
            
            # Ensure the 'answer' key exists and the list is non-empty
            if 'answer' in json_data and isinstance(json_data['answer'], list) and json_data['answer']:
                return json_data
            else:
                # If 'answer' key doesn't exist or the list is empty, return a default value
                print("Warning: 'answer' key is missing or the list is empty.")
                return {"answer": default_value or "No answer available"}
        
        # If no valid JSON boundaries are found, raise an error to go to the except block
        raise ValueError("No JSON object could be found in the response text.")
        
    except json.JSONDecodeError as e:
        print(f"Failed to decode JSON: {e}")  # Log the error
        print(response_text)
        return {"answer": "failed"}  # Default return value when JSON decoding fails
    except Exception as e:
        print(f"An error occurred while loading JSON: {e}")  # Log unexpected errors
        return {"answer": "failed"}  # Default return value for other errors

def save_to_parquet(df, path):
    parent_dir = os.path.dirname(path)
    os.makedirs(parent_dir, exist_ok=True) 
    embedding_cols = [col for col in df.columns if 'embedding' in col.lower()]
    for col in embedding_cols:
        df[col] = df[col].astype(str)
    print(f"Writing df to: {path}")
    df.to_parquet(path)

class RateLimitTracker:
    def __init__(self, request_per_minute=50, tokens_per_minute=40000, tokens_per_day=1000000, buffer_time=5):
        # Initialize rate limit values
        self.REQUESTS_PER_MINUTE = request_per_minute
        self.TOKENS_PER_MINUTE = tokens_per_minute
        self.TOKENS_PER_DAY = tokens_per_day
        self.tokens_used_today = 0
        self.tokens_used_this_minute = 0
        self.last_request_time = time.time()
        self.buffer_time = buffer_time  # Add buffer time to delay requests

    def calculate_delay(self, tokens_used_this_call):
        current_time = time.time()

        # Reset the minute counter if a minute has passed
        if current_time - self.last_request_time > 60:
            self.tokens_used_this_minute = 0

        # Update token usage
        self.tokens_used_today += tokens_used_this_call
        self.tokens_used_this_minute += tokens_used_this_call

        remaining_tokens_minute = self.TOKENS_PER_MINUTE - self.tokens_used_this_minute
        remaining_tokens_day = self.TOKENS_PER_DAY - self.tokens_used_today

        print(f"Debug: Remaining tokens this minute: {remaining_tokens_minute}")
        print(f"Debug: Remaining tokens for the day: {remaining_tokens_day}")

        # If we've used up all tokens in the current minute, wait until the next minute
        if self.tokens_used_this_minute >= self.TOKENS_PER_MINUTE:
            sleep_time = 60 - (current_time - self.last_request_time)
            print(f"Rate limit reached for this minute. Sleeping for {sleep_time + self.buffer_time} seconds.")
            time.sleep(sleep_time + self.buffer_time)  # Add extra buffer time

        # If we've used up all tokens for the day, wait until the next day
        if self.tokens_used_today >= self.TOKENS_PER_DAY:
            time_to_wait = 86400 - (current_time - self.last_request_time)
            print(f"Rate limit reached for today. Sleeping for {time_to_wait + self.buffer_time} seconds.")
            time.sleep(time_to_wait + self.buffer_time)  # Add extra buffer time
            self.tokens_used_today = 0  # Reset after waiting for the full day

        # Update the last request time
        self.last_request_time = time.time()
