from utils import load_data, save_to_parquet
from base import CLIENT
import os
from rag import *
from eval import evaluate_rag_with_ranking

def main():
    run_name = "method-002" # update run name to not overwrite data
    main_outpath = f"{BASE_DIR}/output/main/{run_name}"
    chunkstats_outpath = f"{BASE_DIR}/output/chunk_engineering/{run_name}"
    os.makedirs(main_outpath, exist_ok=True)
    os.makedirs(chunkstats_outpath, exist_ok=True)

    data = load_data()

    available_collections = find_collections(CLIENT)

    df = retrieve_rag_results(available_collections, data, CLIENT)
    save_to_parquet(df, f"{main_outpath}/data.parquet")
    
    summary = evaluate_rag_with_ranking(df)
    summary.to_parquet(f"{chunkstats_outpath}/summary.parquet")

if __name__ == "__main__":
    main()
