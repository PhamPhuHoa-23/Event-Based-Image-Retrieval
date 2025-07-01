import json
import requests
import csv
from typing import Dict, List, Any
from datetime import datetime
import time
import pandas as pd
import argparse

#  Optimal weights:
#    PERSON: 0.500 (was 4.3, -88.4%)
#    CARDINAL: 3.263 (was 3.2, +2.0%)
#    ORG: 0.500 (was 3.8, -86.8%)
#    GPE: 2.658 (was 3.1, -14.3%)
#    EVENT: 4.078 (was 2.9, +40.6%)
#    FAC: 5.000 (was 1.0, +400.0%)
#    NORP: 4.585 (was 1.0, +358.5%)
#    TIME: 2.357 (was 1.0, +135.7%)
#    DATE: 5.000 (was 1.0, +400.0%)
#    PRODUCT: 1.403 (was 1.0, +40.3%)
class QuerySearchSystemEnhanced:
    def __init__(self, es_host="http://localhost:9200", use_private=False, use_clean_articles=False):
        self.es_host = es_host
        self.articles_index = "articles_clean" if use_clean_articles else "articles"
        self.queries_index = "private_queries_clean" if use_private else "queries"
        self.use_private = use_private
        # Analysis
        self.entity_weights = {
            "PERSON": 4.3,      # ↑ Highest frequency (15,114) - players, fans, crowds
            "CARDINAL": 3.5,    # ↑↑ CRITICAL FIX (9,069) - scores, jersey numbers, rankings  
            "ORG": 3.8,         # ↓ Still important but not as frequent
            "GPE": 3.1,         # ↓ Countries, cities, venues
            "EVENT": 2.9,       # ↓ Tournaments, matches, competitions
            "FAC": 2.5,         #  Stadiums, facilities
            "NORP": 2.2,        #  Nationalities, affiliations
            "TIME": 2.1,        # ↑ Match timing, duration
            "DATE": 2.0,        #  Historical context
            "PRODUCT": 2.0,     # ↓ Equipment, brands
            "LAW": 1.8,         # ↓ Rules, regulations
            "LOC": 1.8,         #  General locations  
            "WORK_OF_ART": 1.5, # ↓ Logos, designs
            "MONEY": 1.5,       # ↓ Prize money, transfers
            "PERCENT": 1.5,     # ↑ Statistics, win rates
            "QUANTITY": 1.3,    # ↑ Crowd sizes, distances
            "LANGUAGE": 1.2,    # ↓ Less relevant for visual
            "ORDINAL": 1.2,     # ↑ 1st place, 2nd half
            "MISC": 1.0,        #  Default
            "DEFAULT": 1.0      #  Fallback
        }
        # Reinforcement Learning
        # self.entity_weights = {
        #     "PERSON": 2.539,
        #     "CARDINAL": 2.843,
        #     "ORG": 2.755,
        #     "GPE": 4.487,
        #     "EVENT": 2.567,
        #     "FAC": 2.553,
        #     "NORP": 2.065,
        #     "TIME": 2.323,
        #     "DATE": 3.602,
        #     "PRODUCT": 4.064,
        #     "LAW": 1.777,
        #     "LOC": 2.928,
        #     "WORK_OF_ART": 3.843,
        #     "MONEY": 0.081,
        #     "PERCENT": 3.703,
        #     "QUANTITY": 2.314,
        #     "LANGUAGE": 3.931,
        #     "ORDINAL": 4.215,
        #     "MISC": 0.226,
        #     "DEFAULT": 2.081
        # }
        # Reinforcement Learning - Best Weights:
        #     PERSON: 2.539
        #     CARDINAL: 2.843
        #     ORG: 2.755
        #     GPE: 4.487
        #     EVENT: 2.567
        #     FAC: 2.553
        #     NORP: 2.065
        #     TIME: 2.323
        #     DATE: 3.602
        #     PRODUCT: 4.064
        #     LAW: 1.777
        #     LOC: 2.928
        #     WORK_OF_ART: 3.843
        #     MONEY: 0.081
        #     PERCENT: 3.703
        #     QUANTITY: 2.314
        #     LANGUAGE: 3.931
        #     ORDINAL: 4.215
        #     MISC: 0.226
        #     DEFAULT: 2.081
        # Bayesian Optimizer
        # self.entity_weights = {
        #     "PERSON": 4.5,
        #     "CARDINAL": 3.262762433940483,
        #     "ORG": 3.5,
        #     "GPE": 2.658023721730885,
        #     "EVENT": 4.078077521050077,
        #     "FAC": 5.0,
        #     "NORP": 4.584637762026945,
        #     "TIME": 2.3571108237255967,
        #     "DATE": 5.0,
        #     "PRODUCT": 1.4030332901442306,
        #     "LAW": 1.8,
        #     "LOC": 1.8,
        #     "WORK_OF_ART": 1.5,
        #     "MONEY": 1.5,
        #     "PERCENT": 1.5,
        #     "QUANTITY": 1.3,
        #     "LANGUAGE": 1.2,
        #     "ORDINAL": 1.2,
        #     "MISC": 1.0,
        #     "DEFAULT": 1.0   #  Fallback
        # }
    
    def get_all_queries(self, batch_size=100):
        """Lấy tất cả queries từ DB bằng scroll API"""
        all_queries = []
        
        # Filter for private queries if using private mode
        query_filter = {
            "term": {"data_type": "private"}
        } if self.use_private else {
            "match_all": {}
        }
        
        search_query = {
            "size": batch_size,
            "_source": ["query_id", "query_text", "entities"],
            "query": query_filter,
            "sort": [{"query_id": "asc"}]
        }
        
        try:
            # Initial search với scroll
            response = requests.post(
                f"{self.es_host}/{self.queries_index}/_search?scroll=5m",
                headers={"Content-Type": "application/json"},
                json=search_query
            )
            
            if response.status_code == 200:
                result = response.json()
                scroll_id = result.get("_scroll_id")
                hits = result["hits"]["hits"]
                
                # Process first batch
                for hit in hits:
                    query_data = hit["_source"]
                    all_queries.append({
                        "query_id": query_data["query_id"],
                        "query_text": query_data.get("query_text", ""),
                        "entities": query_data.get("entities", [])
                    })
                
                # Continue scrolling
                while len(hits) > 0:
                    scroll_response = requests.post(
                        f"{self.es_host}/_search/scroll",
                        headers={"Content-Type": "application/json"},
                        json={"scroll": "5m", "scroll_id": scroll_id}
                    )
                    
                    if scroll_response.status_code == 200:
                        scroll_result = scroll_response.json()
                        hits = scroll_result["hits"]["hits"]
                        scroll_id = scroll_result.get("_scroll_id")
                        
                        for hit in hits:
                            query_data = hit["_source"]
                            all_queries.append({
                                "query_id": query_data["query_id"],
                                "query_text": query_data.get("query_text", ""),
                                "entities": query_data.get("entities", [])
                            })
                    else:
                        break
                
                # Clear scroll
                if scroll_id:
                    requests.delete(
                        f"{self.es_host}/_search/scroll",
                        headers={"Content-Type": "application/json"},
                        json={"scroll_id": [scroll_id]}
                    )
                    
        except Exception as e:
            print(f" Lỗi lấy queries: {e}")
            return []
        
        return all_queries
    
    def search_articles_for_query(self, query_entities: List[Dict], top_k=10):
        """Search articles cho một query với weighted entities"""
        if not query_entities:
            return []
        
        should_queries = []
        
        # Tạo weighted queries cho mỗi entity
        for entity in query_entities:
            entity_text = entity.get("text", "").strip()
            entity_label = entity.get("label", "")
            
            if not entity_text:
                continue
                
            # Tính trọng số
            weight = self.entity_weights.get(entity_label, self.entity_weights["DEFAULT"])
            
            # Base query cho entity text
            base_query = {
                "nested": {
                    "path": "entities",
                    "query": {
                        "bool": {
                            "should": [
                                # Exact match với boost cao
                                {
                                    "term": {
                                        "entities.text.keyword": {
                                            "value": entity_text,
                                            "boost": 5.0
                                        }
                                    }
                                },
                                # Fuzzy match
                                {
                                    "match": {
                                        "entities.text": {
                                            "query": entity_text,
                                            "boost": 2.0,
                                            "fuzziness": "AUTO"
                                        }
                                    }
                                },
                                # Prefix match
                                {
                                    "prefix": {
                                        "entities.text.keyword": {
                                            "value": entity_text.lower(),
                                            "boost": 1.5
                                        }
                                    }
                                }
                            ]
                        }
                    },
                    "score_mode": "max"
                }
            }
            
            # Weighted query
            weighted_query = {
                "function_score": {
                    "query": base_query,
                    "boost": weight,
                    "boost_mode": "multiply"
                }
            }
            should_queries.append(weighted_query)
            
            # Bonus query nếu cùng label
            if entity_label:
                same_label_query = {
                    "nested": {
                        "path": "entities",
                        "query": {
                            "bool": {
                                "must": [
                                    {"match": {"entities.text": entity_text}},
                                    {"term": {"entities.label": entity_label}}
                                ]
                            }
                        },
                        "score_mode": "max"
                    }
                }
                
                bonus_weighted_query = {
                    "function_score": {
                        "query": same_label_query,
                        "boost": weight * 1.3,  # Bonus 30% cho cùng label
                        "boost_mode": "multiply"
                    }
                }
                should_queries.append(bonus_weighted_query)
        
        if not should_queries:
            return []
        
        # Main search query
        search_query = {
            "query": {
                "bool": {
                    "should": should_queries,
                    "minimum_should_match": 1
                }
            },
            "size": top_k,
            "_source": ["article_id", "entities"],
            "highlight": {
                "fields": {
                    "entities.text": {}
                }
            }
        }
        
        try:
            response = requests.post(
                f"{self.es_host}/{self.articles_index}/_search",
                headers={"Content-Type": "application/json"},
                json=search_query
            )
            
            if response.status_code == 200:
                result = response.json()
                hits = result["hits"]["hits"]
                
                search_results = []
                for i, hit in enumerate(hits, 1):
                    article_data = hit["_source"]
                    search_results.append({
                        "rank": i,
                        "article_id": article_data["article_id"],
                        "score": hit["_score"],
                        "entities": article_data.get("entities", []),
                        "highlights": hit.get("highlight", {})
                    })
                
                return search_results
            else:
                print(f" Search error: {response.status_code}")
                return []
                
        except Exception as e:
            print(f" Lỗi search: {e}")
            return []
    
    def auto_fill_empty_cells(self, df: pd.DataFrame, top_k: int = 10, fill_value: str = "#") -> pd.DataFrame:
        """
        Auto fill empty cells trong DataFrame với fill_value
        Args:
            df: DataFrame chứa kết quả
            top_k: Số lượng article columns
            fill_value: Giá trị để fill (default: '#')
        """
        df_filled = df.copy()
        
        # Fill NaN values
        df_filled = df_filled.fillna(fill_value)
        
        # Fill empty strings
        df_filled = df_filled.replace('', fill_value)
        
        # Ensure all article_id columns exist and are filled
        for i in range(1, top_k + 1):
            col_name = f"article_id_{i}"
            if col_name not in df_filled.columns:
                df_filled[col_name] = fill_value
            else:
                # Fill any remaining empty values
                df_filled[col_name] = df_filled[col_name].fillna(fill_value)
                df_filled[col_name] = df_filled[col_name].replace('', fill_value)
        
        return df_filled
    
    def search_all_queries_and_save(self, output_submission_csv="submission.csv", 
                                   top_k=10, auto_fill=True, 
                                   max_queries=None, postfix=""):
        """
        Search tất cả queries và save kết quả submission CSV
        Args:
            output_submission_csv: Submission format CSV
            top_k: Số lượng top articles để retrieve (default: 10)
            auto_fill: Tự động fill empty cells (default: True)
            max_queries: Giới hạn số queries để test (default: None = tất cả)
            postfix: Thêm postfix vào tên file
        """
        print(" BẮT ĐẦU ENHANCED QUERY SEARCH")
        print("=" * 50)
        print(f" Top-K Articles: {top_k}")
        print(f" Auto-fill Empty Cells: {auto_fill}")
        print(f" Articles Index: {self.articles_index}")
        print(f" Queries Index: {self.queries_index}")
        if self.use_private:
            print(f" Using PRIVATE queries")
        if max_queries:
            print(f" Max Queries Limit: {max_queries:,}")
        
        start_time = time.time()
        
        # Generate timestamped filenames if postfix not provided
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if postfix:
            output_submission_csv = f"submission_{postfix}.csv"
        else:
            output_submission_csv = f"submission_{timestamp}.csv"
        
        # Lấy tất cả queries
        print(" Loading all queries...")
        all_queries = self.get_all_queries()
        
        if not all_queries:
            print(" Không có queries nào được tìm thấy!")
            return
        
        total_queries = len(all_queries)
        if max_queries and max_queries < total_queries:
            all_queries = all_queries[:max_queries]
            total_queries = max_queries
            print(f" Limited to first {max_queries:,} queries")
        
        print(f" Total queries to process: {total_queries:,}")
        
        # Search từng query
        submission_data = []
        failed_queries = []
        successful_queries = 0
        total_articles_found = 0
        
        for i, query in enumerate(all_queries, 1):
            # Progress tracking
            if i % 50 == 0 or i == total_queries:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                eta = avg_time * (total_queries - i)
                print(f" Progress: {i:,}/{total_queries:,} ({i/total_queries*100:.1f}%) - ETA: {eta/60:.1f}min")
            
            query_id = query["query_id"]
            query_entities = query["entities"]
            
            try:
                # Search articles
                search_results = self.search_articles_for_query(query_entities, top_k=top_k)
                
                if search_results:
                    successful_queries += 1
                    total_articles_found += len(search_results)
                    
                    # Tạo submission row
                    submission_row = {"query_id": query_id}
                    for j in range(1, top_k + 1):
                        if j <= len(search_results):
                            submission_row[f"article_id_{j}"] = search_results[j-1]["article_id"]
                        else:
                            submission_row[f"article_id_{j}"] = ""
                    submission_data.append(submission_row)
                else:
                    # Không có kết quả
                    failed_queries.append(query_id)
                    
                    # Tạo empty submission row
                    submission_row = {"query_id": query_id}
                    for j in range(1, top_k + 1):
                        submission_row[f"article_id_{j}"] = ""
                    submission_data.append(submission_row)
                    
            except Exception as e:
                print(f" Lỗi search query {query_id}: {e}")
                failed_queries.append(query_id)
                
                # Tạo empty submission row cho failed query
                submission_row = {"query_id": query_id}
                for j in range(1, top_k + 1):
                    submission_row[f"article_id_{j}"] = ""
                submission_data.append(submission_row)
        
        total_time = time.time() - start_time
        
        # Save submission CSV với auto-fill
        try:
            submission_df = pd.DataFrame(submission_data)
            
            # Ensure proper column order
            fieldnames = ["query_id"] + [f"article_id_{i}" for i in range(1, top_k + 1)]
            submission_df = submission_df.reindex(columns=fieldnames, fill_value="")
            
            # Auto-fill empty cells if requested
            if auto_fill:
                submission_df = self.auto_fill_empty_cells(submission_df, top_k, fill_value="#")
                print(f" Auto-filled empty cells with '#'")
            
            submission_df.to_csv(output_submission_csv, index=False)
            print(f" Saved submission CSV: {output_submission_csv}")
            
            # Verification
            empty_cells = (submission_df == "").sum().sum() if not auto_fill else (submission_df == "#").sum().sum()
            total_cells = submission_df.size
            print(f" Empty/filled cells: {empty_cells:,}/{total_cells:,} ({empty_cells/total_cells*100:.1f}%)")
            
        except Exception as e:
            print(f" Lỗi save submission CSV: {e}")
        
        # Print statistics
        self._print_search_statistics(total_queries, successful_queries, total_articles_found, failed_queries, total_time, top_k)
        
        return {
            "submission_data": submission_data,
            "files": {
                "submission_csv": output_submission_csv
            }
        }
    
    def _print_search_statistics(self, total_queries: int, successful_queries: int, total_articles_found: int, 
                               failed_queries: List, total_time: float, top_k: int = 10):
        """Print search statistics"""
        print("\n ENHANCED SEARCH STATISTICS:")
        print("=" * 40)
        
        print(f" Total queries: {total_queries:,}")
        print(f" Có kết quả: {successful_queries:,} ({successful_queries/total_queries*100:.1f}%)")
        print(f" Không có kết quả: {total_queries - successful_queries:,}")
        print(f" Total articles found: {total_articles_found:,}")
        print(f" Avg articles per successful query: {total_articles_found/successful_queries:.1f}" if successful_queries > 0 else " Avg articles per successful query: 0")
        print(f" Top-K setting: {top_k}")
        print(f"⏱  Thời gian: {total_time/60:.1f} phút")
        print(f" Tốc độ: {total_queries/total_time:.1f} queries/giây")
        
        if failed_queries:
            print(f"  Failed queries: {len(failed_queries)} (examples: {failed_queries[:5]})")
    
    def search_sample_queries(self, sample_size=10, top_k=10):
        """Search một số queries mẫu để test"""
        print(f" Testing với {sample_size} queries mẫu (Top-{top_k})")
        print("=" * 40)
        print(f" Articles Index: {self.articles_index}")
        print(f" Queries Index: {self.queries_index}")
        if self.use_private:
            print(f" Using PRIVATE queries")
        
        all_queries = self.get_all_queries()
        if not all_queries:
            print(" Không có queries nào!")
            return
        
        # Lấy sample
        sample_queries = all_queries[:sample_size]
        
        for i, query in enumerate(sample_queries, 1):
            query_id = query["query_id"]
            query_entities = query["entities"]
            
            print(f"\n Query {i}: {query_id}")
            print(f" Entities: {len(query_entities)}")
            
            # Show top 3 entities
            if query_entities:
                top_entities = [f"{e.get('text', '')} ({e.get('label', '')})" for e in query_entities[:3]]
                print(f"  Top entities: {', '.join(top_entities)}")
            
            # Search
            results = self.search_articles_for_query(query_entities, top_k=top_k)
            
            if results:
                print(f" Found {len(results)} articles")
                for j, result in enumerate(results[:3], 1):  # Show top 3
                    print(f"   {j}. {result['article_id']} (score: {result['score']:.2f})")
            else:
                print(" Không tìm thấy articles")

def main():
    parser = argparse.ArgumentParser(description="Enhanced Query Search System")
    
    # Required arguments
    parser.add_argument("--mode", choices=["search_all", "sample"], default="search_all",
                       help="Search mode: search_all hoặc sample")
    
    # Optional arguments
    parser.add_argument("--es_host", default="http://localhost:9200", help="Elasticsearch host")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top articles to retrieve (default: 10)")
    parser.add_argument("--max_queries", type=int, help="Max number of queries to process (for testing)")
    parser.add_argument("--sample_size", type=int, default=10, help="Sample size for testing mode")
    parser.add_argument("--postfix", default="", help="Postfix for output filenames")
    parser.add_argument("--no_auto_fill", action="store_true", help="Disable auto-fill of empty cells with '#'")
    parser.add_argument("--private", action="store_true", help="Use private_queries_clean instead of queries")
    parser.add_argument("--clean_articles", action="store_true", help="Use articles_clean instead of articles")
    
    args = parser.parse_args()
    
    print(" ENHANCED QUERY SEARCH SYSTEM")
    print("=" * 40)
    print(f"Mode: {args.mode}")
    print(f"ES Host: {args.es_host}")
    print(f"Top-K Articles: {args.top_k}")
    print(f"Use Private Queries: {args.private}")
    print(f"Use Clean Articles: {args.clean_articles}")
    if args.max_queries:
        print(f"Max Queries: {args.max_queries:,}")
    print(f"Auto-fill Empty Cells: {not args.no_auto_fill}")
    
    # Initialize system
    try:
        searcher = QuerySearchSystemEnhanced(
            es_host=args.es_host,
            use_private=args.private,
            use_clean_articles=args.clean_articles
        )
        
        if args.mode == "search_all":
            # Search all queries
            result = searcher.search_all_queries_and_save(
                top_k=args.top_k,
                auto_fill=not args.no_auto_fill,
                max_queries=args.max_queries,
                postfix=args.postfix
            )
            
            if result:
                print(f"\n HOÀN THÀNH!")
                print(f" File created: {result['files']['submission_csv']}")
        
        elif args.mode == "sample":
            # Test với sample queries
            searcher.search_sample_queries(
                sample_size=args.sample_size,
                top_k=args.top_k
            )
        
    except Exception as e:
        print(f" Lỗi: {e}")
        exit(1)

if __name__ == "__main__":
    main() 