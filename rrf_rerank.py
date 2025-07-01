#!/usr/bin/env python3
"""
RRF Reranking Script for submission files

Usage: python rrf_rerank.py <postfix1> <postfix2> [postfix3] [postfix4] ... [--k K] [--top-n N] [--adaptive]

Examples:
  python rrf_rerank.py baseline SNEKCAVE_20 bayes10
  python rrf_rerank.py baseline SNEKCAVE_20 bayes10 --k 60 --top-n 10
  python rrf_rerank.py BayesFull Bayes ReWeighted --top-n 20
  python rrf_rerank.py model1 model2 model3 model4 model5 --k 100
  python rrf_rerank.py baseline enhanced --adaptive  # Use adaptive mode

Features:
- Automatically detects number of article columns in each file
- Supports both article_id_ and image_id_ column formats  
- Auto-detects output size from input files (or specify with --top-n)
- Handles missing articles (marked with # or empty)
- Two RRF modes:
  * NORMAL MODE (default): Anti-bias - skips queries empty in ANY input file
  * ADAPTIVE MODE (--adaptive): Dynamic filtering - với mỗi query, tìm số lượng 
    article ít nhất trong các file, sau đó chỉ xét gấp đôi số lượng đó cho re-ranking
- Supports 2+ input files (flexible number of submissions)
"""

import pandas as pd
import sys
from datetime import datetime
import os
from collections import defaultdict
import argparse

def rrf_score(rank, k=60):
    """
    Calculate RRF score for a given rank
    RRF(d) = sum(1/(k + rank_i)) for all systems i where document d appears
    """
    if rank == 0:  # Article not found in this ranking
        return 0
    return 1 / (k + rank)

def load_submission_file(postfix):
    """Load submission file with given postfix"""
    filepath = f"submission_{postfix}.csv"
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File {filepath} not found")
    
    print(f"Loading {filepath}...")
    df = pd.read_csv(filepath)
    return df

def perform_rrf_reranking_adaptive(postfixes, k=60, top_n=None):
    """
    Perform RRF reranking với adaptive filtering - mỗi query sẽ tìm số lượng article 
    ít nhất trong các file, sau đó chỉ xét gấp đôi số lượng đó cho re-ranking
    
    Args:
        postfixes: List of postfixes for submission files
        k: RRF parameter (default: 60)
        top_n: Number of top articles to output (None = auto-detect from input files)
    
    Returns:
        Tuple: (DataFrame with reranked results, skipped_queries_count)
    """
    # Load all submission files
    dfs = []
    article_col_counts = []
    
    for postfix in postfixes:
        df = load_submission_file(postfix)
        dfs.append(df)
        
        # Count article columns in this file
        article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
        article_col_counts.append(len(article_cols))
        print(f"  File submission_{postfix}.csv has {len(article_cols)} article columns")
    
    # Determine output size
    if top_n is None:
        top_n = max(article_col_counts)
        print(f"Auto-detected output size: {top_n} articles per query")
    else:
        print(f"Using specified output size: {top_n} articles per query")
    
    # Get all unique query_ids
    all_query_ids = set()
    for df in dfs:
        all_query_ids.update(df['query_id'].tolist())
    
    all_query_ids = sorted(list(all_query_ids))
    
    # Process each query with adaptive filtering
    reranked_results = []
    total_queries = len(all_query_ids)
    skipped_queries = 0
    
    print(f" Adaptive RRF mode: Each query uses dynamic article count based on minimum available")
    
    for idx, query_id in enumerate(all_query_ids, 1):
        if idx % 100 == 0 or idx == total_queries:
            print(f"Processing query {idx}/{total_queries}: {query_id}")
        
        # First pass: Find minimum valid article count across all files for this query
        query_article_counts = []
        valid_files = []
        
        for file_idx, df in enumerate(dfs):
            query_row = df[df['query_id'] == query_id]
            
            if query_row.empty:
                query_article_counts.append(0)
                valid_files.append(False)
                continue
                
            query_row = query_row.iloc[0]
            
            # Count valid articles in this file for this query
            article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
            article_cols = sorted(article_cols, key=lambda x: int(x.split('_')[-1]))
            
            valid_article_count = 0
            for col in article_cols:
                article_id = query_row[col]
                if not (pd.isna(article_id) or str(article_id).strip() == '#' or str(article_id).strip() == ''):
                    valid_article_count += 1
                else:
                    break  # Stop at first empty/# article to get consecutive count
            
            query_article_counts.append(valid_article_count)
            valid_files.append(valid_article_count > 0)
        
        # Check if query has results in all files
        if not all(valid_files):
            skipped_queries += 1
            if idx <= 10:
                empty_files = [f"submission_{postfixes[i]}.csv" for i, is_valid in enumerate(valid_files) if not is_valid]
                print(f"   Skipping query {query_id} - empty/missing in: {', '.join(empty_files)}")
            
            # Create empty result row
            result_row = {'query_id': query_id}
            for i in range(top_n):
                result_row[f'article_id_{i+1}'] = '#'
            reranked_results.append(result_row)
            continue
        
        # Find minimum article count and calculate dynamic limit
        min_article_count = min(query_article_counts)
        dynamic_limit = min(min_article_count * 2, max(query_article_counts))  # Gấp đôi nhưng không vượt quá max available
        
        if idx <= 10:  # Log first few queries for debugging
            print(f"   Query {query_id}: article counts = {query_article_counts}, min = {min_article_count}, using limit = {dynamic_limit}")
        
        # Second pass: Collect articles with dynamic limit
        article_scores = defaultdict(float)
        
        for file_idx, df in enumerate(dfs):
            query_row = df[df['query_id'] == query_id].iloc[0]
            
            # Extract article columns
            article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
            article_cols = sorted(article_cols, key=lambda x: int(x.split('_')[-1]))
            
            # Use only first dynamic_limit articles from this file
            articles_used = 0
            for rank, col in enumerate(article_cols, 1):
                if articles_used >= dynamic_limit:
                    break
                    
                article_id = query_row[col]
                
                # Skip if article is missing
                if pd.isna(article_id) or str(article_id).strip() == '#' or str(article_id).strip() == '':
                    continue
                
                # Add RRF score for this article
                article_scores[article_id] += rrf_score(rank, k)
                articles_used += 1
        
        # Sort articles by RRF score (descending)
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create result row
        result_row = {'query_id': query_id}
        
        # Fill top articles
        for i in range(top_n):
            if i < len(sorted_articles):
                result_row[f'article_id_{i+1}'] = sorted_articles[i][0]
            else:
                result_row[f'article_id_{i+1}'] = '#'
        
        reranked_results.append(result_row)
    
    # Create result DataFrame
    result_df = pd.DataFrame(reranked_results)
    
    # Sort by query_id to maintain order
    result_df = result_df.sort_values('query_id').reset_index(drop=True)
    
    print(f" Adaptive RRF summary: {skipped_queries}/{total_queries} queries skipped ({skipped_queries/total_queries*100:.1f}%)")
    
    return result_df, skipped_queries

def perform_rrf_reranking(postfixes, k=60, top_n=None):
    """
    Perform RRF reranking on multiple submission files
    
    Args:
        postfixes: List of postfixes for submission files
        k: RRF parameter (default: 60)
        top_n: Number of top articles to output (None = auto-detect from input files)
    
    Returns:
        Tuple: (DataFrame with reranked results, skipped_queries_count)
    """
    # Load all submission files
    dfs = []
    article_col_counts = []
    
    for postfix in postfixes:
        df = load_submission_file(postfix)
        dfs.append(df)
        
        # Count article columns in this file
        article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
        article_col_counts.append(len(article_cols))
        print(f"  File submission_{postfix}.csv has {len(article_cols)} article columns")
    
    # Determine output size
    if top_n is None:
        # Use the maximum number of columns from all files
        top_n = max(article_col_counts)
        print(f"Auto-detected output size: {top_n} articles per query")
    else:
        print(f"Using specified output size: {top_n} articles per query")
    
    # Get all unique query_ids
    all_query_ids = set()
    for df in dfs:
        all_query_ids.update(df['query_id'].tolist())
    
    all_query_ids = sorted(list(all_query_ids))
    
    # Process each query
    reranked_results = []
    total_queries = len(all_query_ids)
    skipped_queries = 0
    
    print(f" Anti-bias mode: Queries empty in ANY file will be skipped")
    
    for idx, query_id in enumerate(all_query_ids, 1):
        if idx % 100 == 0 or idx == total_queries:
            print(f"Processing query {idx}/{total_queries}: {query_id}")
        
        # Check if query exists and has valid articles in ALL files
        query_has_results_in_all_files = True
        file_has_valid_articles = []
        
        for file_idx, df in enumerate(dfs):
            query_row = df[df['query_id'] == query_id]
            
            if query_row.empty:
                # Query doesn't exist in this file
                query_has_results_in_all_files = False
                file_has_valid_articles.append(False)
                continue
                
            query_row = query_row.iloc[0]
            
            # Check if this file has any valid articles for this query
            article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
            article_cols = sorted(article_cols, key=lambda x: int(x.split('_')[-1]))
            
            has_valid_articles = False
            for col in article_cols:
                article_id = query_row[col]
                if not (pd.isna(article_id) or str(article_id).strip() == '#' or str(article_id).strip() == ''):
                    has_valid_articles = True
                    break
            
            file_has_valid_articles.append(has_valid_articles)
            if not has_valid_articles:
                query_has_results_in_all_files = False
        
        # Skip query if it's empty in ANY file (avoid bias)
        if not query_has_results_in_all_files:
            skipped_queries += 1
            if idx <= 10:  # Log first few skipped queries
                empty_files = [f"submission_{postfixes[i]}.csv" for i, has_results in enumerate(file_has_valid_articles) if not has_results]
                print(f"   Skipping query {query_id} - empty/missing in: {', '.join(empty_files)}")
            
            # Create empty result row
            result_row = {'query_id': query_id}
            for i in range(top_n):
                result_row[f'article_id_{i+1}'] = '#'
            reranked_results.append(result_row)
            continue
        
        # Collect all articles and their ranks from all systems (only if query valid in all files)
        article_scores = defaultdict(float)
        
        for df in dfs:
            query_row = df[df['query_id'] == query_id]
            query_row = query_row.iloc[0]
            
            # Extract article columns (support both article_id_ and image_id_ formats)
            article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
            article_cols = sorted(article_cols, key=lambda x: int(x.split('_')[-1]))  # Sort by number
            
            for rank, col in enumerate(article_cols, 1):
                article_id = query_row[col]
                
                # Skip if article is missing (marked with # or NaN)
                if pd.isna(article_id) or str(article_id).strip() == '#' or str(article_id).strip() == '':
                    continue
                    
                # Add RRF score for this article
                article_scores[article_id] += rrf_score(rank, k)
        
        # Sort articles by RRF score (descending)
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Create result row
        result_row = {'query_id': query_id}
        
        # Fill top articles
        for i in range(top_n):
            if i < len(sorted_articles):
                result_row[f'article_id_{i+1}'] = sorted_articles[i][0]
            else:
                result_row[f'article_id_{i+1}'] = '#'
        
        reranked_results.append(result_row)
    
    # Create result DataFrame
    result_df = pd.DataFrame(reranked_results)
    
    # Sort by query_id to maintain order
    result_df = result_df.sort_values('query_id').reset_index(drop=True)
    
    print(f" Anti-bias summary: {skipped_queries}/{total_queries} queries skipped ({skipped_queries/total_queries*100:.1f}%)")
    
    return result_df, skipped_queries

def main():
    parser = argparse.ArgumentParser(description='RRF Reranking for submission files')
    parser.add_argument('postfixes', nargs='+', help='Postfixes for submission files (minimum 2 required)')
    parser.add_argument('--k', type=int, default=60, help='RRF parameter k (default: 60)')
    parser.add_argument('--top-n', type=int, default=None, help='Number of top articles to output (default: auto-detect from input files)')
    parser.add_argument('--adaptive', action='store_true', help='Use adaptive RRF mode (dynamic filtering based on minimum article count)')
    parser.add_argument('--normal', action='store_true', default=True, help='Use normal RRF mode (traditional anti-bias approach, default)')
    
    args = parser.parse_args()
    
    # Validate minimum number of files
    if len(args.postfixes) < 2:
        print(" Error: At least 2 submission files are required for RRF reranking")
        sys.exit(1)
    
    # Determine RRF mode
    adaptive_mode = args.adaptive
    
    print("=== RRF Reranking Script ===")
    print(f"Input files ({len(args.postfixes)}): {[f'submission_{p}.csv' for p in args.postfixes]}")
    print(f"RRF parameter k: {args.k}")
    print(f"RRF mode: {'ADAPTIVE (dynamic filtering)' if adaptive_mode else 'NORMAL (anti-bias)'}")
    if args.top_n:
        print(f"Output size: {args.top_n} articles per query")
    else:
        print("Output size: auto-detect from input files")
    
    try:
        # Perform RRF reranking
        if adaptive_mode:
            result_df, skipped_queries = perform_rrf_reranking_adaptive(args.postfixes, args.k, args.top_n)
            mode_suffix = "adaptive"
        else:
            result_df, skipped_queries = perform_rrf_reranking(args.postfixes, args.k, args.top_n)
            mode_suffix = "normal"
        
        # Create output directory
        os.makedirs('ReRank', exist_ok=True)
        
        # Generate output filename with current datetime and mode
        current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"ReRank/sb_rrank_{mode_suffix}_{current_datetime}.csv"
        
        # Save result
        result_df.to_csv(output_filename, index=False)
        
        print(f"\n=== Results ===")
        print(f"Total queries processed: {len(result_df)}")
        print(f"Queries with valid RRF: {len(result_df) - skipped_queries}")
        print(f"Queries skipped (anti-bias): {skipped_queries} ({skipped_queries/len(result_df)*100:.1f}%)")
        print(f"Output saved to: {output_filename}")
        print(f"File size: {os.path.getsize(output_filename)} bytes")
        
        # Show sample of results
        print(f"\n=== Sample Results ===")
        print(result_df.head())
        
        # Show statistics about input files
        print(f"\n=== Input File Statistics ===")
        for postfix in args.postfixes:
            filepath = f"submission_{postfix}.csv"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                article_cols = [col for col in df.columns if col.startswith('article_id_') or col.startswith('image_id_')]
                print(f"  {filepath}: {len(df)} queries, {len(article_cols)} article columns")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        print("\nAvailable submission files:")
        submission_files = [f for f in os.listdir('.') if f.startswith('submission_') and f.endswith('.csv')]
        for f in sorted(submission_files):
            postfix = f.replace('submission_', '').replace('.csv', '')
            print(f"  - {postfix}")
    else:
        main() 