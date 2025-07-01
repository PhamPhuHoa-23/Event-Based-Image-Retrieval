#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Professional Search Pipeline: Enhanced Entity Search + Multi-Model Image Search

Complete end-to-end search pipeline with:
1. Enhanced entity search with weighted entity scoring for text retrieval
2. Multi-model image search with article ranking priority  
3. Configurable model weights and parameters (JSON config support)
4. Flexible CSV re-ranking support
5. Voting and RRF aggregation modes

JSON Config Format:
{
  "<database_name>": {
    "weight": <database_family_weight>,
    "query_collections": [
      {"<query_collection_name>": <collection_weight>}, ...
    ]
  }
}

Example:
{
  "Database_Initialized_Large": {
    "weight": 1.0,
    "query_collections": [
      {"Query_Initialized_Large": 1.0},
      {"Summary_Initialized_Large": 0.8},
      {"Concise_Initialized_Large": 1.2}
    ]
  }
}

Usage:
  # JSON config with voting mode
  python search_pipeline.py --json-config config.json --use-voting
  
  # JSON config with RRF mode (default)
  python search_pipeline.py --json-config config.json --use-rrf
  
  # Legacy mode with individual parameters
  python search_pipeline.py --primary-query-large-weight 1.0 --use-voting
  
  # Text search only (enhanced entity search)
  python search_pipeline.py --text-search-only --json-config config.json
  
  # Image search only from existing CSV files  
  python search_pipeline.py --image-search-only --csv-files submission1.csv submission2.csv --json-config config.json --use-voting
  
  # Complete end-to-end pipeline with JSON config
  python search_pipeline.py --json-config config.json --use-voting --debug
"""

import sys
import os
import json
import csv
import pandas as pd
import argparse
import time
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

# Import từ existing modules
from entity_search_system import QuerySearchSystemEnhanced

try:
    from qdrant_client import QdrantClient
    from qdrant_client.models import Filter, FieldCondition, MatchAny, QueryRequest
except ImportError:
    print(" Cần cài qdrant-client: pip install qdrant-client")
    sys.exit(1)

try:
    from rrf_rerank import perform_rrf_reranking_adaptive, load_submission_file
except ImportError:
    print(" Không thể import rrf_rerank module. Đảm bảo file rrf_rerank.py ở cùng thư mục.")

class ProfessionalSearchPipeline:
    def __init__(self, 
                 # Text search config
                 es_host="http://localhost:9200",
                 expansion_file="query_expansion.json",
                 blacklist_file="celeb_names.txt",
                 use_clean_index=False,
                 articles_by_year_file="articles_by_year.json",  # NEW: Date filtering
                 
                 # Image search config
                 qdrant_host="localhost", 
                 qdrant_port=6333, 
                 qdrant_url=None,
                 article_mapping_json="database_article_to_images_v.0.1.json",
                 
                 # Private test mode
                 private_test_mode=False,  # Add Private_ prefix to query collections
                 
                 # JSON Config Support
                 json_config_file=None,  # NEW: JSON config file path
                 
                 # Legacy: Checkpoint/Dataset Selection (deprecated when using JSON config)
                 primary_checkpoint="Initialized",  # Main checkpoint: "Initialized", "Flickr30k", "OpenEvents_v1"
                 enable_h14_laion=True,  # Enable H14 Laion as additional model
                 
                 # Legacy: Multi-Model Support (deprecated when using JSON config)
                 enable_multi_model=True,  # Enable multi-model search
                 
                 # Legacy: Primary Checkpoint Weights (deprecated when using JSON config)
                 primary_query_large_weight=1.0,
                 primary_summary_large_weight=1.0,
                 primary_concise_large_weight=1.2,
                 
                 # Legacy: Primary Checkpoint Weights (Base Model)
                 primary_query_base_weight=0.9,
                 primary_summary_base_weight=0.9,
                 primary_concise_base_weight=1.1,
                 
                 # Legacy: H14 Laion weights (deprecated when using JSON config)
                 h14_query_weight=0.8,
                 h14_summary_weight=0.8,
                 h14_concise_weight=1.0,
                 
                 # Legacy: Family weights (deprecated when using JSON config)
                 primary_large_family_weight=1.0,
                 primary_base_family_weight=0.9,
                 h14_laion_family_weight=0.8,
                 
                 # Search config
                 article_ranking_boost=0.3,  # Boost cho article ranking
                 rrf_k=60,
                 multi_model_rrf_k=50,  # RRF k for multi-model reranking
                 use_voting=False,  # NEW: Use voting instead of RRF
                 
                 # Advanced Sigmoid Boosting (Optimized Config: Balanced Penalty)
                 use_sigmoid_boosting=True,
                 similarity_threshold=0.6,  # Optimized: Ngưỡng similarity để boost
                 similarity_weight=10.0,    # Optimized: Trọng số cho similarity trong sigmoid
                 rank_weight=2.5,           # Optimized: Balanced penalty cho rank (2.0→2.5)
                 sigmoid_bias=0.0,          # Optimized: Bias cho sigmoid function
                 max_boost_factor=0.5,      # Optimized: Max boost có thể áp dụng
                 
                 debug=False):
        
        # Store search configuration
        self.use_voting = use_voting
        
        # Always store legacy configuration (for backward compatibility)
        self.primary_checkpoint = primary_checkpoint
        self.enable_h14_laion = enable_h14_laion
        self.enable_multi_model = enable_multi_model
        
        # Validate checkpoint selection
        valid_checkpoints = ["Initialized", "Flickr30k", "OpenEvents_v1"]
        if primary_checkpoint not in valid_checkpoints:
            raise ValueError(f"Invalid checkpoint. Must be one of: {valid_checkpoints}")
        
        # Load JSON config or use legacy parameters
        if json_config_file:
            print(f" Loading JSON config from: {json_config_file}")
            self.config_data = self.load_json_config(json_config_file)
            self.enable_multi_model = True  # JSON config always uses multi-model mode
        else:
            print(" Using legacy parameter configuration")
            self.config_data = None
        
        # Database mapping: OpenEvents_v1 searches on Flickr30k database
        self.database_mapping = {
            "Initialized": "Initialized",
            "Flickr30k": "Flickr30k", 
            "OpenEvents_v1": "Flickr30k"  # OpenEvents_v1 searches on Flickr30k database
        }
        
        # Text search system - Enhanced Entity Search System
        self.text_system = QuerySearchSystemEnhanced(
            es_host=es_host,
            use_private=private_test_mode,  # Use private queries if in private test mode
            use_clean_articles=use_clean_index  # Use clean articles index if specified
        )
        
        # Image search setup
        try:
            if qdrant_url:
                self.client = QdrantClient(url=qdrant_url)
                print(f" Connected to Qdrant: {qdrant_url}")
            else:
                self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
                print(f" Connected to Qdrant: {qdrant_host}:{qdrant_port}")
        except Exception as e:
            print(f" Lỗi kết nối Qdrant: {e}")
            raise
        
        # Load article mapping
        self.article_to_images = self.load_article_mapping(article_mapping_json)
        
        # Private test mode configuration
        self.private_test_mode = private_test_mode
        
        # Build model configuration
        if self.config_data:
            # Use JSON config
            self.build_config_from_json()
        else:
            # Use legacy configuration
            self._build_legacy_config(
                primary_checkpoint, enable_h14_laion, enable_multi_model,
                primary_query_large_weight, primary_summary_large_weight, primary_concise_large_weight,
                primary_query_base_weight, primary_summary_base_weight, primary_concise_base_weight,
                h14_query_weight, h14_summary_weight, h14_concise_weight,
                primary_large_family_weight, primary_base_family_weight, h14_laion_family_weight,
                private_test_mode
            )
        
        # Search config
        self.article_ranking_boost = article_ranking_boost
        self.rrf_k = rrf_k
        self.multi_model_rrf_k = multi_model_rrf_k
        self.debug = debug
        
        # Advanced Sigmoid Boosting config
        self.use_sigmoid_boosting = use_sigmoid_boosting
        self.similarity_threshold = similarity_threshold
        self.similarity_weight = similarity_weight
        self.rank_weight = rank_weight
        self.sigmoid_bias = sigmoid_bias
        self.max_boost_factor = max_boost_factor
        
        # Debug counter for limiting debug output
        self._debug_query_count = 0
        self._max_debug_queries = 1  # Only show debug for first query
        
        # Filter active collections/families based on weights
        if enable_multi_model:
            # Filter active families with weight > 0
            self.active_families = {
                name: config for name, config in self.model_families.items() 
                if config["family_weight"] > 0.0
            }
            # Check for filtered families
            filtered_families = {
                name: config for name, config in self.model_families.items() 
                if config["family_weight"] <= 0.0
            }
            if filtered_families:
                print(f" Filtered out families with weight=0: {list(filtered_families.keys())}")
        else:
            # Filter active collections with weight > 0 (single-model mode)
            self.active_model_weights = {
                name: weight for name, weight in self.model_weights.items() 
                if weight > 0.0
            }
            # Check for filtered collections
            filtered_collections = {
                name: weight for name, weight in self.model_weights.items() 
                if weight <= 0.0
            }
            if filtered_collections:
                print(f" Filtered out collections with weight=0: {list(filtered_collections.keys())}")
        
        # Validation: Ensure at least one collection/family is active
        if enable_multi_model:
            if not self.active_families:
                raise ValueError(" All model families have weight=0. At least one family must have weight > 0.")
        else:
            if not self.active_model_weights:
                raise ValueError(" All model collections have weight=0. At least one collection must have weight > 0.")
        
        # Print configuration
        if self.config_data:
            # JSON config mode
            print(f" JSON CONFIG MODE")
            print(f"    Database configurations: {len(self.config_data)}")
            if hasattr(self, 'model_families'):
                print(f"    Model families: {len(self.model_families)}")
                print(f"    Active families: {list(self.active_families.keys())}")
            if hasattr(self, 'model_weights'):
                print(f"    Total model weights: {len(self.model_weights)}")
        else:
            # Legacy mode
            database_name = self.database_mapping[self.primary_checkpoint]
            
            if self.enable_multi_model:
                family_count = len(self.active_families) if hasattr(self, 'active_families') else 0
                print(f" LEGACY MULTI-MODEL MODE ({family_count} families)")
                print(f"    Primary checkpoint: {self.primary_checkpoint}")
                print(f"    Database mapping: {self.primary_checkpoint} → {database_name}")
                if hasattr(self, 'active_families'):
                    print(f"    Active families: {list(self.active_families.keys())}")
            else:
                print(f" LEGACY SINGLE-MODEL MODE ({self.primary_checkpoint}-Large only)")
                print(f"    Primary checkpoint: {self.primary_checkpoint}")
                print(f"    Database mapping: {self.primary_checkpoint} → {database_name}")
                if hasattr(self, 'active_model_weights'):
                    print(f"    Active collections: {list(self.active_model_weights.keys())}")
        
        # Print aggregation mode
        aggregation_mode = "VOTING" if self.use_voting else "RRF"
        print(f" AGGREGATION MODE: {aggregation_mode}")
        if not self.use_voting:
            print(f"    RRF k: {self.rrf_k}, Multi-model RRF k: {self.multi_model_rrf_k}")
        
        # Print date filtering status
        print(f" DATE FILTERING: DISABLED (using original cascade system)")
        
        # Print private test mode status
        if self.private_test_mode:
            print(f" PRIVATE TEST MODE: ENABLED (using Private_ prefix for query collections)")
        else:
            print(f" PRIVATE TEST MODE: DISABLED (using standard query collections)")
        
        print(f" Article ranking boost: {self.article_ranking_boost}")
        if self.use_sigmoid_boosting:
            print(f" Using Sigmoid Boosting:")
            print(f"    Similarity threshold: {self.similarity_threshold}")
            print(f"    Similarity weight: {self.similarity_weight}, Rank weight: {self.rank_weight}")
            print(f"    Max boost factor: {self.max_boost_factor}")
    
    def load_json_config(self, json_config_file: str) -> Dict:
        """
        Load JSON config file với format:
        {
           "<database_name>": {
              "weight": float,
              "query_collections": [
                 {"<query-collection-name>": <weight>}, ...
              ]
           }
        }
        """
        try:
            with open(json_config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            print(f" Loaded JSON config:")
            for db_name, db_config in config.items():
                db_weight = db_config.get("weight", 1.0)
                query_collections = db_config.get("query_collections", [])
                print(f"    Database: {db_name} (weight: {db_weight})")
                for q_col in query_collections:
                    for col_name, col_weight in q_col.items():
                        print(f"       {col_name}: {col_weight}")
            
            return config
        except Exception as e:
            print(f" Error loading JSON config: {e}")
            raise
    
    def build_config_from_json(self):
        """Build model weights and families from JSON config"""
        if not self.config_data:
            return
        
        # Build model families from JSON config
        self.model_families = {}
        self.model_weights = {}
        
        for db_name, db_config in self.config_data.items():
            db_weight = db_config.get("weight", 1.0)
            query_collections = db_config.get("query_collections", [])
            
            # Extract query collection names and weights
            query_col_names = []
            for q_col in query_collections:
                for col_name, col_weight in q_col.items():
                    # Apply private prefix if needed
                    full_col_name = f"Private_{col_name}" if self.private_test_mode else col_name
                    query_col_names.append(full_col_name)
                    self.model_weights[full_col_name] = col_weight
            
            # Create family configuration
            family_config = {
                "query_collections": query_col_names,
                "search_collection": db_name,  # Database name is the search collection
                "family_weight": db_weight
            }
            
            # Use full database name as family name (để tránh overwrite)
            family_name = db_name.replace("Database_", "")
            self.model_families[family_name] = family_config
        
        print(f" Built from JSON config:")
        print(f"    Model families: {len(self.model_families)}")
        print(f"    Model weights: {len(self.model_weights)}")
    
    def _build_legacy_config(self, primary_checkpoint, enable_h14_laion, enable_multi_model,
                           primary_query_large_weight, primary_summary_large_weight, primary_concise_large_weight,
                           primary_query_base_weight, primary_summary_base_weight, primary_concise_base_weight,
                           h14_query_weight, h14_summary_weight, h14_concise_weight,
                           primary_large_family_weight, primary_base_family_weight, h14_laion_family_weight,
                           private_test_mode):
        """Build legacy configuration from individual parameters"""
        self.enable_multi_model = enable_multi_model
        
        if enable_multi_model:
            # Build model weights based on primary checkpoint and H14 Laion
            base_model_weights = {}
            
            # Primary checkpoint collections (Large)
            base_model_weights[f"Query_{primary_checkpoint}_Large"] = primary_query_large_weight
            base_model_weights[f"Summary_{primary_checkpoint}_Large"] = primary_summary_large_weight
            base_model_weights[f"Concise_{primary_checkpoint}_Large"] = primary_concise_large_weight
            
            # Primary checkpoint collections (Base)
            base_model_weights[f"Query_{primary_checkpoint}_Base"] = primary_query_base_weight
            base_model_weights[f"Summary_{primary_checkpoint}_Base"] = primary_summary_base_weight
            base_model_weights[f"Concise_{primary_checkpoint}_Base"] = primary_concise_base_weight
            
            # H14 Laion collections (if enabled)
            if enable_h14_laion:
                base_model_weights["Query_Laion_H14"] = h14_query_weight
                base_model_weights["Summary_Laion_H14"] = h14_summary_weight
                base_model_weights["Concise_Laion_H14"] = h14_concise_weight
            
            # Apply private prefix to model weights
            self.model_weights = {}
            for base_name, weight in base_model_weights.items():
                collection_name = f"Private_{base_name}" if private_test_mode else base_name
                self.model_weights[collection_name] = weight
            
            # Build family definitions based on primary checkpoint and H14 Laion
            base_families = {}
            
            # Primary checkpoint families - with correct database mapping
            database_name = self.database_mapping[primary_checkpoint]
            
            primary_large_family = {
                "query_collections": [f"Query_{primary_checkpoint}_Large", f"Summary_{primary_checkpoint}_Large", f"Concise_{primary_checkpoint}_Large"],
                "search_collection": f"Database_{database_name}_Large",
                "family_weight": primary_large_family_weight
            }
            
            primary_base_family = {
                "query_collections": [f"Query_{primary_checkpoint}_Base", f"Summary_{primary_checkpoint}_Base", f"Concise_{primary_checkpoint}_Base"],
                "search_collection": f"Database_{database_name}_Base",
                "family_weight": primary_base_family_weight
            }
            
            base_families[f"{primary_checkpoint}-Large"] = primary_large_family
            base_families[f"{primary_checkpoint}-Base"] = primary_base_family
            
            # H14 Laion family (if enabled)
            if enable_h14_laion:
                h14_family = {
                    "query_collections": ["Query_Laion_H14", "Summary_Laion_H14", "Concise_Laion_H14"],
                    "search_collection": "Database_Laion_H14",
                    "family_weight": h14_laion_family_weight
                }
                base_families["H14-Laion"] = h14_family
            
            # Apply private prefix to family query collections
            self.model_families = {}
            for family_name, family_config in base_families.items():
                updated_config = family_config.copy()
                updated_config["query_collections"] = [
                    f"Private_{col}" if private_test_mode else col 
                    for col in family_config["query_collections"]
                ]
                # Search collections don't get prefix
                self.model_families[family_name] = updated_config
        else:
            # Single-model mode (primary checkpoint Large only)
            base_collections = {
                f"Query_{primary_checkpoint}_Large": primary_query_large_weight,
                f"Summary_{primary_checkpoint}_Large": primary_summary_large_weight,
                f"Concise_{primary_checkpoint}_Large": primary_concise_large_weight
            }
            
            # Apply private prefix to model weights
            self.model_weights = {}
            for base_name, weight in base_collections.items():
                collection_name = f"Private_{base_name}" if private_test_mode else base_name
                self.model_weights[collection_name] = weight
            
            self.model_families = None
    
    def create_output_directory(self, config_name=None):
        """Tạo output directory trong app_results"""
        if config_name:
            dir_name = config_name
        else:
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            dir_name = f"pipeline_{current_datetime}"
        
        output_dir = os.path.join("app_results", dir_name)
        os.makedirs(output_dir, exist_ok=True)
        
        print(f" Output directory: {output_dir}")
        return output_dir
    
    def save_config(self, output_dir, args=None, **kwargs):
        """Save pipeline config to text file"""
        config_file = os.path.join(output_dir, "config.txt")
        
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write("=== END-TO-END SEARCH PIPELINE CONFIG ===\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Save all argparse arguments if provided
            if args is not None:
                f.write("=== ALL ARGUMENTS ===\n")
                args_dict = vars(args)
                for key, value in sorted(args_dict.items()):
                    f.write(f"{key}: {value}\n")
                f.write("\n")
            
            f.write("=== ACTIVE MODEL WEIGHTS ===\n")
            for model, weight in self.model_weights.items():
                f.write(f"{model}: {weight}\n")
            
            f.write(f"\n=== PIPELINE INTERNAL CONFIG ===\n")
            f.write(f"Primary checkpoint: {self.primary_checkpoint}\n")
            f.write(f"Enable H14 Laion: {self.enable_h14_laion}\n")
            f.write(f"Private test mode: {self.private_test_mode}\n")
            f.write(f"Database mapping: {self.database_mapping}\n")
            f.write(f"Article ranking boost: {self.article_ranking_boost}\n")
            f.write(f"RRF k: {self.rrf_k}\n")
            f.write(f"Multi-model RRF k: {self.multi_model_rrf_k}\n")
            f.write(f"Use sigmoid boosting: {self.use_sigmoid_boosting}\n")
            if self.use_sigmoid_boosting:
                f.write(f"Similarity threshold: {self.similarity_threshold}\n")
                f.write(f"Similarity weight: {self.similarity_weight}\n")
                f.write(f"Rank weight: {self.rank_weight}\n")
                f.write(f"Sigmoid bias: {self.sigmoid_bias}\n")
                f.write(f"Max boost factor: {self.max_boost_factor}\n")
            
            # Save additional pipeline parameters if any
            if kwargs:
                f.write(f"\n=== ADDITIONAL PIPELINE PARAMETERS ===\n")
                for key, value in kwargs.items():
                    f.write(f"{key}: {value}\n")
        
        print(f" Config saved: {config_file}")
        return config_file
    
    def load_article_mapping(self, article_mapping_json: str) -> Dict[str, List[str]]:
        """Load article to images mapping"""
        try:
            with open(article_mapping_json, 'r', encoding='utf-8') as f:
                mapping = json.load(f)
            print(f" Loaded {len(mapping):,} article mappings")
            return mapping
        except Exception as e:
            print(f" Lỗi đọc article mapping: {e}")
            return {}
    
    def text_search_pipeline(self, output_dir=None, filename_suffix="cascade", top_k=10, max_queries=None) -> Tuple[str, str]:
        """
        Chạy text search pipeline với cascade + celebrity filtering
        Returns: (csv_path, json_path)
        """
        print("\n TEXT SEARCH PIPELINE (CASCADE + CELEBRITY)")
        print("=" * 60)
        
        if output_dir:
            # Save directly to output directory
            csv_path = os.path.join(output_dir, f"submission_{filename_suffix}.csv")
            json_path = os.path.join(output_dir, f"stage_1_{filename_suffix}.json")
            
            # Enhanced entity search with weighted entity scoring
            results = self.text_system.search_all_queries_and_save(
                output_submission_csv=csv_path,
                top_k=top_k,
                auto_fill=True,
                max_queries=max_queries,
                postfix=f"_{filename_suffix}"
            )
            
            # Create JSON file with search results details
            import json
            json_data = {
                "search_results": results.get("submission_data", []),
                "files": results.get("files", {}),
                "pipeline_info": {
                    "search_type": "enhanced_entity_search",
                    "top_k": top_k,
                    "max_queries": max_queries,
                    "filename_suffix": filename_suffix,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            try:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, ensure_ascii=False, indent=2)
                print(f" Created JSON: {json_path}")
            except Exception as e:
                print(f" Could not create JSON file: {e}")
                json_path = None
        else:
            # Enhanced entity search with weighted entity scoring
            results = self.text_system.search_all_queries_and_save(
                output_submission_csv=f"submission_{filename_suffix}.csv",
                top_k=top_k,
                auto_fill=True,
                max_queries=max_queries,
                postfix=f"_{filename_suffix}"
            )
            
            csv_path = f"submission_{filename_suffix}.csv"
            json_path = f"stage_1_{filename_suffix}.json"  # Updated for consistency
        
        print(f" Text search completed:")
        print(f"    CSV: {csv_path}")
        print(f"    JSON: {json_path}")
        return csv_path, json_path
    
    def rrf_rerank_csvs(self, csv_files: List[str], adaptive_mode=True, output_dir=None) -> str:
        """
        RRF rerank CSV files (1 file cũng được - sẽ copy trực tiếp)
        Returns: path to RRF CSV file
        """
        print(f"\n PROCESSING {len(csv_files)} CSV FILE(S)")
        print("=" * 60)
        
        if len(csv_files) == 1:
            # Chỉ có 1 file - copy trực tiếp
            csv_file = csv_files[0]
            print(f" Single file mode: {csv_file}")
            
            # Copy file to output directory
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"submission_single_file.csv")
                
                # Read and save to maintain consistency
                import pandas as pd
                df = pd.read_csv(csv_file)
                # Fill NaN values với '#' để đảm bảo consistency
                df = df.fillna('#')
                df.to_csv(output_path, index=False)
                
                print(f" File copied: {output_path}")
                return output_path
            else:
                print(f" Using original file: {csv_file}")
                return csv_file
        
        # Multiple files - RRF
        # Copy files to current directory with proper naming for RRF system
        postfixes = []
        copied_files = []
        
        for i, csv_file in enumerate(csv_files):
            # Ensure file exists
            if not os.path.exists(csv_file):
                print(f" File not found: {csv_file}")
                continue
                
            basename = os.path.basename(csv_file)
            
            # Generate postfix
            if basename.startswith("submission_") and basename.endswith(".csv"):
                postfix = basename.replace("submission_", "").replace(".csv", "")
            else:
                # Use filename as postfix for non-standard names
                postfix = basename.replace(".csv", "").replace("submission_", "")
                # Clean postfix (remove special chars)
                postfix = "".join(c for c in postfix if c.isalnum() or c in "._-")
            
            # Ensure unique postfix
            original_postfix = postfix
            counter = 1
            while postfix in postfixes:
                postfix = f"{original_postfix}_{counter}"
                counter += 1
            
            postfixes.append(postfix)
            target_file = f"submission_{postfix}.csv"  # Target in working directory
            
            # Always copy file to working directory for RRF processing
            try:
                shutil.copy2(csv_file, target_file)
                copied_files.append(target_file)
                print(f" Copied: {csv_file} -> {target_file}")
            except Exception as e:
                print(f" Failed to copy {csv_file}: {e}")
                continue
        
        if not postfixes:
            raise ValueError("No valid CSV files found")
        
        print(f" RRF Processing postfixes: {postfixes}")
        
        try:
            # Thực hiện RRF
            if adaptive_mode:
                result_df, skipped_queries = perform_rrf_reranking_adaptive(postfixes, self.rrf_k)
                mode_suffix = "adaptive"
            else:
                # Import normal RRF if needed
                from rrf_rerank import perform_rrf_reranking
                result_df, skipped_queries = perform_rrf_reranking(postfixes, self.rrf_k)
                mode_suffix = "normal"
        finally:
            # Clean up copied files
            for copied_file in copied_files:
                try:
                    os.remove(copied_file)
                    print(f" Cleaned up: {copied_file}")
                except Exception as e:
                    print(f" Could not remove {copied_file}: {e}")
        
        # Save RRF result
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            rrf_csv_path = os.path.join(output_dir, f"submission_rrf_{mode_suffix}.csv")
        else:
            os.makedirs('ReRank', exist_ok=True)
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            rrf_csv_path = f"ReRank/e2e_rrf_{mode_suffix}_{current_datetime}.csv"
        
        # Fill NaN values với '#' trước khi lưu
        result_df = result_df.fillna('#')
        result_df.to_csv(rrf_csv_path, index=False)
        
        print(f" RRF completed: {rrf_csv_path}")
        print(f" Queries processed: {len(result_df)}, Skipped: {skipped_queries}")
        
        return rrf_csv_path
    
    def multi_model_image_search_pipeline(self, text_search_csv: str,
                                         max_articles_per_query=15,
                                         direct_search_top_k=20,
                                         final_top_k=10) -> Dict[str, List[str]]:
        """
        Multi-model image search pipeline với 3 model families
        Returns: final_results dict
        """
        if not self.enable_multi_model:
            raise ValueError("Multi-model mode not enabled. Use image_search_pipeline() instead.")
        
        print(f"\n MULTI-MODEL IMAGE SEARCH PIPELINE")
        print("=" * 60)
        print(f" Text search CSV: {text_search_csv}")
        
        # Use pre-filtered active families
        active_families = self.active_families
        print(f" Active model families: {list(active_families.keys())}")
        print(f" Max articles per query: {max_articles_per_query}")
        print(f" Direct search top-k: {direct_search_top_k}")
        print(f" Final top-k: {final_top_k}")
        
        # Load và classify queries
        query_to_articles, queries_with_articles, queries_without_articles = self.load_and_classify_queries(
            text_search_csv, max_articles_per_query
        )
        
        print(f" Queries với articles: {len(queries_with_articles)}")
        print(f" Queries không có articles: {len(queries_without_articles)}")
        
        # Results từ tất cả model families
        family_results = {}
        
        # Search với từng model family
        for family_name, family_config in active_families.items():
            print(f"\n Processing {family_name} family...")
            
            query_collections = family_config["query_collections"]
            search_collection = family_config["search_collection"]
            family_weight = family_config["family_weight"]
            
            print(f"    Query collections: {query_collections}")
            print(f"    Search collection: {search_collection}")
            print(f"    Family weight: {family_weight}")
            
            # Search results cho family này
            family_search_results = {}
            
            # 1. Queries CÓ articles
            if queries_with_articles:
                results_with_articles = self.search_queries_with_articles(
                    queries_with_articles, query_to_articles, query_collections, 
                    search_collection, final_top_k
                )
                family_search_results.update(results_with_articles)
            
            # 2. Queries KHÔNG có articles
            if queries_without_articles:
                results_without_articles = self.search_queries_without_articles(
                    queries_without_articles, query_collections, search_collection,
                    direct_search_top_k, final_top_k
                )
                family_search_results.update(results_without_articles)
            
            # Aggregate within family collections
            if self.use_voting:
                family_final_results = self.voting_final_collections(family_search_results)
            else:
                family_final_results = self.rrf_final_collections(family_search_results)
            
            #  MEMORY OPTIMIZATION: Save intermediate results and clear variables
            import tempfile
            import pickle
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{family_name}.pkl")
            with open(temp_file.name, 'wb') as f:
                pickle.dump(family_final_results, f)
            
            family_results[family_name] = {
                "results": family_final_results,
                "weight": family_weight,
                "temp_file": temp_file.name  # Store temp file path
            }
            
            # Clear large variables to save memory
            del family_search_results
            if 'results_with_articles' in locals():
                del results_with_articles
            if 'results_without_articles' in locals():
                del results_without_articles
            
            print(f" {family_name} completed: {len(family_final_results)} queries (temp saved)")
            print(f" Temp file: {temp_file.name}")
        
        # Multi-model aggregation across families
        if self.use_voting:
            print(f"\n MULTI-MODEL VOTING across {len(family_results)} families...")
            final_results = self.multi_model_rrf(family_results, final_top_k, use_voting=True)
        else:
            print(f"\n MULTI-MODEL RRF across {len(family_results)} families...")
            final_results = self.multi_model_rrf(family_results, final_top_k, use_voting=False)
        
        #  CLEANUP: Remove temp files
        print(f"\n Cleaning up temp files...")
        for family_name, family_data in family_results.items():
            if "temp_file" in family_data:
                try:
                    import os
                    os.unlink(family_data["temp_file"])
                    print(f"    Removed {family_data['temp_file']}")
                except Exception as e:
                    print(f"    Could not remove {family_data['temp_file']}: {e}")
        
        print(f" Multi-model image search completed: {len(final_results)} queries processed")
        return final_results
    
    def get_query_collection_name(self, base_name: str) -> str:
        """Helper method to get correct collection name based on private test mode"""
        if self.private_test_mode:
            return f"Private_{base_name}"
        else:
            return f"Public_{base_name}"
    
    def image_search_pipeline(self, text_search_csv: str, 
                            query_collections=None, 
                            search_collection=None,
                            max_articles_per_query=15,
                            direct_search_top_k=20,
                            final_top_k=10) -> Dict[str, List[str]]:
        """
        Image search pipeline với article ranking priority
        """
        if query_collections is None:
            # Build default query collections with private/public prefix
            query_collections = [
                self.get_query_collection_name(f"Query_{self.primary_checkpoint}_Large"), 
                self.get_query_collection_name(f"Summary_{self.primary_checkpoint}_Large"), 
                self.get_query_collection_name(f"Concise_{self.primary_checkpoint}_Large")
            ]
        
        if search_collection is None:
            # Build default search collection with correct database mapping
            database_name = self.database_mapping[self.primary_checkpoint]
            search_collection = f"Database_{database_name}_Large"
        
        # Filter query_collections to only include active ones (weight > 0)
        if hasattr(self, 'active_model_weights'):
            query_collections = [col for col in query_collections if col in self.active_model_weights]
            print(f" Filtered active query collections: {query_collections}")
        
        if not query_collections:
            print(" No active query collections available (all weights are 0)")
            return {}
        
        print(f"\n IMAGE SEARCH PIPELINE")
        print("=" * 60)
        print(f" Text search CSV: {text_search_csv}")
        print(f" Query collections: {query_collections}")
        print(f" Search collection: {search_collection}")
        print(f" Max articles per query: {max_articles_per_query}")
        print(f" Direct search top-k: {direct_search_top_k}")
        print(f" Final top-k: {final_top_k}")
        
        # Load text search results và phân loại queries
        query_to_articles, queries_with_articles, queries_without_articles = self.load_and_classify_queries(
            text_search_csv, max_articles_per_query
        )
        
        print(f" Queries với articles: {len(queries_with_articles)}")
        print(f" Queries không có articles: {len(queries_without_articles)}")
        
        # Search cho từng loại query
        all_search_results = {}
        
        # 1. Queries CÓ articles - search với article filtering và ranking boost
        if queries_with_articles:
            print(f"\n Processing queries WITH articles...")
            results_with_articles = self.search_queries_with_articles(
                queries_with_articles, query_to_articles, query_collections, 
                search_collection, final_top_k
            )
            all_search_results.update(results_with_articles)
        
        # 2. Queries KHÔNG có articles - search trực tiếp với Query-BEiT3-Large
        if queries_without_articles:
            print(f"\n Processing queries WITHOUT articles...")
            results_without_articles = self.search_queries_without_articles(
                queries_without_articles, query_collections, search_collection,
                direct_search_top_k, final_top_k
            )
            all_search_results.update(results_without_articles)
        
        # Aggregate final results từ multiple collections
        if self.use_voting:
            print(f"\n Final VOTING across collections...")
            final_results = self.voting_final_collections(all_search_results)
        else:
            print(f"\n Final RRF across collections...")
            final_results = self.rrf_final_collections(all_search_results)
        
        print(f" Image search completed: {len(final_results)} queries processed")
        return final_results
    
    def load_and_classify_queries(self, csv_path: str, max_articles: int) -> Tuple[Dict, List, List]:
        """Load CSV và phân loại queries có/không có articles"""
        query_to_articles = {}
        queries_with_articles = []
        queries_without_articles = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                query_id = row['query_id']
                article_ids = []
                
                # Extract article_ids
                i = 1
                while f'article_id_{i}' in row:
                    article_id = row[f'article_id_{i}'].strip()
                    if article_id and article_id != '#':
                        article_ids.append(article_id)
                    i += 1
                    
                    # Giới hạn số lượng
                    if len(article_ids) >= max_articles:
                        break
                
                query_to_articles[query_id] = article_ids
                
                if article_ids:
                    queries_with_articles.append(query_id)
                else:
                    queries_without_articles.append(query_id)
        
        return query_to_articles, queries_with_articles, queries_without_articles
    
    def search_queries_with_articles(self, queries: List[str], query_to_articles: Dict,
                                   query_collections: List[str], search_collection: str, 
                                   top_k: int) -> Dict[str, Dict[str, List[Dict]]]:
        """Search cho queries có articles với article ranking boost"""
        all_results = {}
        
        for query_idx, query_id in enumerate(queries, 1):
            if query_idx % 50 == 0 or query_idx == len(queries):
                print(f" Processing query {query_idx}/{len(queries)}: {query_id}")
            
            article_ids = query_to_articles[query_id]
            
            # Get candidate images từ articles
            candidate_images = []
            article_rank_map = {}  # Map image_id -> article_rank
            
            for rank, article_id in enumerate(article_ids, 1):
                if article_id in self.article_to_images:
                    for image_id in self.article_to_images[article_id]:
                        candidate_images.append(image_id)
                        article_rank_map[image_id] = rank
            
            if not candidate_images:
                continue
            
            # Remove duplicates while preserving order
            seen = set()
            unique_candidates = []
            for img_id in candidate_images:
                if img_id not in seen:
                    unique_candidates.append(img_id)
                    seen.add(img_id)
            
            # Search trên multiple collections (only active ones)
            query_results = {}
            active_collections = query_collections
            
            # Filter collections với active weights (trong multi-model hoặc single-model)
            if hasattr(self, 'active_model_weights'):
                active_collections = [col for col in query_collections if col in self.active_model_weights]
            elif hasattr(self, 'active_families'):
                # Multi-model mode: get active collections from active families
                active_collections = []
                for family_config in self.active_families.values():
                    active_collections.extend([col for col in family_config["query_collections"] if col in query_collections])
            
            # Debug: Print query processing info
            if self.debug:
                print(f"\n DEBUG: Processing query {query_id} WITH articles")
                print(f" Articles: {len(article_ids)} | Candidate images: {len(unique_candidates)}")
                print(f" Active collections: {active_collections}")
            
            for collection_name in active_collections:
                query_vector = self.get_query_embedding(collection_name, query_id)
                if query_vector:
                    results = self.search_similar_images_with_ranking_boost(
                        search_collection, query_vector, unique_candidates, 
                        article_rank_map, top_k * 2  # Get more for better ranking
                    )
                    query_results[collection_name] = results
                else:
                    query_results[collection_name] = []
            
            all_results[query_id] = query_results
        
        return all_results
    
    def search_similar_images_with_ranking_boost(self, collection_name: str, 
                                               query_vector: List[float],
                                               candidate_image_ids: List[str],
                                               article_rank_map: Dict[str, int],
                                               top_k: int = 20) -> List[Dict]:
        """Search với article ranking boost"""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="image_id",
                            match=MatchAny(any=candidate_image_ids)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True,
                score_threshold=0.0
            )
            
            # Debug: Print similarity scores
            if self.debug and len(search_result) > 0:
                similarities = [hit.score for hit in search_result]
                print(f"\n DEBUG: Search results from {collection_name}")
                print(f" Candidates: {len(candidate_image_ids)}, Found: {len(search_result)}")
                print(f" Similarity range: {min(similarities):.6f} - {max(similarities):.6f} | Avg: {sum(similarities)/len(similarities):.6f}")
                print(" Top similarity scores:")
                for i, hit in enumerate(search_result[:5], 1):
                    image_id = hit.payload.get("image_id", "unknown")
                    similarity = hit.score
                    article_rank = article_rank_map.get(image_id, 999)
                    print(f"   {i}. {image_id} | Similarity: {similarity:.6f} | Article rank: #{article_rank}")
                print()
            
            results = []
            for hit in search_result:
                image_id = hit.payload.get("image_id", "unknown")
                base_score = hit.score  # This is the similarity score from Qdrant
                
                #  Advanced Sigmoid Boosting với similarity + article rank
                article_rank = article_rank_map.get(image_id, 999)
                ranking_boost = self.calculate_sigmoid_boost(base_score, article_rank)
                
                final_score = base_score + ranking_boost
                
                results.append({
                    "image_id": image_id,
                    "score": final_score,
                    "base_score": base_score,
                    "ranking_boost": ranking_boost,
                    "article_rank": article_rank,
                    "boost_explanation": self.get_boost_explanation(base_score, article_rank, ranking_boost),
                    "payload": hit.payload
                })
            
            # Debug output for first query with results (show detailed scoring)
            if self.debug and results and len(results) > 0 and self._debug_query_count < self._max_debug_queries:
                self._debug_query_count += 1
                print(f"\n DEBUG: Sigmoid Boosting Details for collection '{collection_name}' (Query #{self._debug_query_count})")
                if self.use_sigmoid_boosting:
                    print(f" Sigmoid Config: sim_weight={self.similarity_weight}, rank_weight={self.rank_weight}, max_boost={self.max_boost_factor}")
                else:
                    print(f" Simple boosting: factor={self.article_ranking_boost}")
                print(f" Showing top 5 results:")
                print("-" * 100)
                for i, result in enumerate(results[:5], 1):
                    print(f"{i:2d}. Image: {result['image_id']}")
                    print(f"     Base similarity: {result['base_score']:.6f}")
                    print(f"     Article rank: #{result['article_rank']}")
                    print(f"     {result['boost_explanation']}")
                    print(f"     Final score: {result['score']:.6f}")
                    print()
            
            # Sort by final score
            results.sort(key=lambda x: x["score"], reverse=True)
            
            return results
            
        except Exception as e:
            print(f" Search error: {e}")
            return []
    
    def search_queries_without_articles(self, queries: List[str], 
                                      query_collections: List[str],
                                      search_collection: str,
                                      direct_search_top_k: int,
                                      final_top_k: int) -> Dict[str, Dict[str, List[Dict]]]:
        """Search cho queries không có articles - OPTIMIZE: chỉ dùng Summary và Concise"""
        all_results = {}
        
        # OPTIMIZE: Bỏ Query- collection, chỉ dùng Summary và Concise
        optimized_collections = [col for col in query_collections if not col.startswith("Query-")]
        
        # Filter collections với active weights (trong multi-model hoặc single-model)
        if hasattr(self, 'active_model_weights'):
            optimized_collections = [col for col in optimized_collections if col in self.active_model_weights]
        elif hasattr(self, 'active_families'):
            # Multi-model mode: get active collections from active families
            active_collections_from_families = []
            for family_config in self.active_families.values():
                active_collections_from_families.extend(family_config["query_collections"])
            optimized_collections = [col for col in optimized_collections if col in active_collections_from_families]
        
        print(f" OPTIMIZED: Using {optimized_collections} for {len(queries)} queries without articles")
        
        if not optimized_collections:
            print(" No active optimized collections available (all weights are 0)")
            return {}
        
        for query_idx, query_id in enumerate(queries, 1):
            if query_idx % 50 == 0 or query_idx == len(queries):
                print(f" Processing query {query_idx}/{len(queries)}: {query_id}")
            
            query_results = {}
            
            # Debug: Print query processing info for queries without articles
            if self.debug:
                print(f"\n DEBUG: Processing query {query_id} WITHOUT articles")
                print(f" Optimized collections: {optimized_collections}")
            
            # Chỉ search với optimized collections (đã được filter active)
            for collection_name in optimized_collections:
                query_vector = self.get_query_embedding(collection_name, query_id)
                if query_vector:
                    # Search trực tiếp trong database collection
                    results = self.search_similar_images_no_filter(
                        search_collection, query_vector, direct_search_top_k
                    )
                    query_results[collection_name] = results
                else:
                    query_results[collection_name] = []
            
            all_results[query_id] = query_results
        
        return all_results
    
    def get_query_embedding(self, collection_name: str, query_id: str) -> Optional[List[float]]:
        """Get query embedding từ Qdrant collection"""
        try:
            search_result = self.client.scroll(
                collection_name=collection_name,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="image_id",
                            match=MatchAny(any=[query_id])
                        )
                    ]
                ),
                limit=1,
                with_payload=True,
                with_vectors=True
            )
            
            if search_result[0]:
                point = search_result[0][0]
                if self.debug:
                    print(f" Found embedding for query {query_id} in {collection_name} | Vector dim: {len(point.vector) if point.vector else 0}")
                return point.vector
            else:
                if self.debug:
                    print(f" No embedding found for query {query_id} in {collection_name}")
                return None
                
        except Exception as e:
            if self.debug:
                print(f" Error getting embedding for {query_id} from {collection_name}: {e}")
            return None
    
    def search_similar_images_no_filter(self, collection_name: str, 
                                      query_vector: List[float], 
                                      top_k: int = 50) -> List[Dict]:
        """Search không filter - cho queries không có articles"""
        try:
            search_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                with_payload=True,
                score_threshold=0.0
            )
            
            # Debug: Print similarity scores for direct search
            if self.debug and len(search_result) > 0:
                similarities = [hit.score for hit in search_result]
                print(f"\n DEBUG: Direct search results from {collection_name}")
                print(f" Found: {len(search_result)} results")
                print(f" Similarity range: {min(similarities):.6f} - {max(similarities):.6f} | Avg: {sum(similarities)/len(similarities):.6f}")
                print(" Top similarity scores:")
                for i, hit in enumerate(search_result[:5], 1):
                    image_id = hit.payload.get("image_id", "unknown")
                    similarity = hit.score
                    print(f"   {i}. {image_id} | Similarity: {similarity:.6f}")
                print()
            
            results = []
            for rank, hit in enumerate(search_result, 1):
                results.append({
                    "rank": rank,
                    "image_id": hit.payload.get("image_id", "unknown"),
                    "score": hit.score,
                    "payload": hit.payload
                })
            
            return results
            
        except Exception as e:
            print(f" Direct search error: {e}")
            return []
    
    def rrf_final_collections(self, search_results: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, List[str]]:
        """RRF kết quả từ multiple collections với model weights (chỉ active collections)"""
        return self._aggregate_final_collections(search_results, use_voting=False)
    
    def voting_final_collections(self, search_results: Dict[str, Dict[str, List[Dict]]]) -> Dict[str, List[str]]:
        """Voting kết quả từ multiple collections với model weights (chỉ active collections)"""
        return self._aggregate_final_collections(search_results, use_voting=True)
    
    def _aggregate_final_collections(self, search_results: Dict[str, Dict[str, List[Dict]]], use_voting: bool = False) -> Dict[str, List[str]]:
        """Aggregate kết quả từ multiple collections với RRF hoặc Voting"""
        # Determine active weights based on mode
        if hasattr(self, 'active_model_weights'):
            active_weights = self.active_model_weights
        else:
            # Multi-model mode: collect weights from active families
            active_weights = {}
            if hasattr(self, 'active_families'):
                for family_config in self.active_families.values():
                    for col in family_config["query_collections"]:
                        if col in self.model_weights:
                            active_weights[col] = self.model_weights[col]
            else:
                # Fallback to all weights
                active_weights = self.model_weights
        
        mode_name = "VOTING" if use_voting else "RRF"
        print(f" {mode_name} with active model weights: {active_weights}")
        
        final_results = {}
        
        for query_id, collection_results in search_results.items():
            # Aggregate scores cho mỗi image từ multiple collections (chỉ active)
            image_scores = defaultdict(float)
            
            for collection_name, results in collection_results.items():
                # Chỉ xử lý collections có weight > 0
                weight = active_weights.get(collection_name, 0.0)
                if weight <= 0.0:
                    continue
                
                for rank, result in enumerate(results, 1):
                    image_id = result["image_id"]
                    
                    if use_voting:
                        # Voting: mỗi collection vote với weight, không quan tâm rank
                        vote_score = weight
                    else:
                        # RRF score với weight
                        vote_score = weight / (self.rrf_k + rank)
                    
                    image_scores[image_id] += vote_score
            
            # Sort by aggregated score
            sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Top images
            final_results[query_id] = [img_id for img_id, score in sorted_images[:50]]
        
        return final_results
    
    def multi_model_rrf(self, family_results: Dict[str, Dict], final_top_k: int = 50, use_voting: bool = False) -> Dict[str, List[str]]:
        """
        Multi-model RRF/Voting across different model families
        
        Args:
            family_results: {family_name: {"results": Dict[query_id, List[image_ids]], "weight": float}}
            final_top_k: Final top-k images per query
            use_voting: Use voting instead of RRF
            
        Returns: final_results dict
        """
        mode_name = "VOTING" if use_voting else "RRF"
        print(f" Multi-model {mode_name} with {len(family_results)} families:")
        for family_name, data in family_results.items():
            weight = data["weight"]
            num_queries = len(data["results"])
            print(f"    {family_name}: {num_queries} queries, weight={weight}")
        
        # Get all unique query_ids
        all_query_ids = set()
        for family_data in family_results.values():
            all_query_ids.update(family_data["results"].keys())
        
        final_results = {}
        
        for query_id in all_query_ids:
            # Aggregate scores across families
            image_scores = defaultdict(float)
            
            for family_name, family_data in family_results.items():
                family_weight = family_data["weight"]
                family_query_results = family_data["results"].get(query_id, [])
                
                # Score cho family này
                for rank, image_id in enumerate(family_query_results, 1):
                    if rank <= final_top_k:  # Only consider top-k from each family
                        if use_voting:
                            # Voting: family vote với weight, không quan tâm rank
                            score = family_weight
                        else:
                            # RRF score với weight
                            score = family_weight / (self.multi_model_rrf_k + rank)
                        image_scores[image_id] += score
            
            # Sort by aggregated score
            sorted_images = sorted(image_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Final top-k images
            final_results[query_id] = [img_id for img_id, score in sorted_images[:final_top_k]]
        
        print(f" Multi-model {mode_name} completed: {len(final_results)} queries processed")
        return final_results
    
    def save_final_image_results(self, final_results: Dict[str, List[str]], output_dir=None, filename_suffix="") -> str:
        """Save final image search results as track2_<suffix>.csv"""
        if output_dir:
            output_path = os.path.join(output_dir, f"track2_{filename_suffix}.csv")
        else:
            os.makedirs('ReRank', exist_ok=True)
            current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"ReRank/e2e_final_images_{current_datetime}.csv"
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['query_id'] + [f'image_id_{i}' for i in range(1, 11)]
            writer.writerow(header)
            
            # Data
            for query_id in sorted(final_results.keys()):
                row = [query_id]
                images = final_results[query_id]
                
                # Pad với # cho các ô trống
                for i in range(50):
                    if i < len(images):
                        row.append(images[i])
                    else:
                        row.append('#')
                
                writer.writerow(row)
        
        print(f" Saved {len(final_results)} query results to: {output_path}")
        return output_path
    
    def run_integrated_cascade_pipeline(self, config_name=None, 
                                       text_top_k=10, max_queries=None,
                                       additional_csv_files=None, adaptive_rrf=True,
                                       args=None) -> Tuple[str, str, str]:
        """
        Run complete integrated cascade pipeline
        
        Args:
            config_name: Name for output directory (or datetime if None)
            text_top_k: Top-k for text search
            max_queries: Limit queries for testing
            additional_csv_files: Optional list of additional CSV files for RRF with cascade results
            adaptive_rrf: Use adaptive RRF mode
            
        Returns: (submission_csv_path, stage_1_json_path, track2_csv_path)
        """
        print(" INTEGRATED CASCADE + IMAGE SEARCH PIPELINE")
        print("=" * 80)
        start_time = time.time()
        
        # Create output directory
        output_dir = self.create_output_directory(config_name)
        
        # Save config with all args
        config_params = {
            "text_top_k": text_top_k,
            "max_queries": max_queries,
            "additional_csv_files": additional_csv_files,
            "adaptive_rrf": adaptive_rrf,
            "pipeline_mode": "integrated_cascade"
        }
        self.save_config(output_dir, args, **config_params)
        
        # Step 1: ALWAYS run CASCADE text search first
        print(" Running CASCADE text search...")
        cascade_csv_path, stage_1_json_path = self.text_search_pipeline(
            output_dir, "cascade", text_top_k, max_queries
        )
        
        # Step 2: Optional RRF with additional CSV files
        if additional_csv_files:
            print(f" RRF CASCADE results with {len(additional_csv_files)} additional files...")
            
            # Combine cascade result with additional files
            all_csv_files = [cascade_csv_path] + additional_csv_files
            
            submission_csv_path = self.rrf_rerank_csvs(all_csv_files, adaptive_rrf, output_dir)
            print(f" RRF completed: CASCADE + {additional_csv_files}")
        else:
            print(" Using CASCADE results directly (no additional RRF)")
            submission_csv_path = cascade_csv_path
        
        # Step 2: Image search (multi-model or single-model)
        if self.enable_multi_model:
            print(" Running MULTI-MODEL image search...")
            final_results = self.multi_model_image_search_pipeline(
                submission_csv_path,
                max_articles_per_query=15,
                direct_search_top_k=20,
                final_top_k=15  # Increased from 10 to 15 for less aggressive filtering
            )
        else:
            print(" Running SINGLE-MODEL image search...")
            final_results = self.image_search_pipeline(
                submission_csv_path,
                max_articles_per_query=15,
                direct_search_top_k=20,
                final_top_k=15  # Increased from 10 to 15 for less aggressive filtering
            )
        
        # Save track2 results
        filename_suffix = config_name if config_name else "pipeline"
        track2_csv_path = self.save_final_image_results(
            final_results, output_dir, filename_suffix
        )
        
        total_time = time.time() - start_time
        
        print("\n END-TO-END PIPELINE COMPLETED!")
        print("=" * 80)
        print(f"⏱ Total time: {total_time/60:.1f} minutes")
        print(f" Output directory: {output_dir}")
        print(f" Submission CSV: {submission_csv_path}")
        if stage_1_json_path:
            print(f" Stage 1 JSON: {stage_1_json_path}")
        print(f" Track 2 CSV: {track2_csv_path}")
        
        return submission_csv_path, stage_1_json_path, track2_csv_path
    
    def calculate_sigmoid_boost(self, similarity_score: float, article_rank: int) -> float:
        """
        Advanced sigmoid boosting mechanism kết hợp similarity và article rank
        
        Logic:
        - High similarity (>0.8) + any rank -> significant boost
        - Low similarity (<0.6) + any rank -> minimal/no boost  
        - Medium similarity (0.6-0.8) + good rank -> moderate boost
        
        Formula: sigmoid(similarity_weight * similarity - rank_weight * log(rank) + bias) * max_boost
        """
        import math
        
        if not self.use_sigmoid_boosting:
            # Fallback to simple boosting
            return self.article_ranking_boost / article_rank
        
        # Hard threshold: không boost nếu similarity quá thấp
        if similarity_score < 0.5:
            return 0.0
        
        # Sigmoid input calculation
        # Positive factors: high similarity
        # Negative factors: high rank (log scale to reduce impact)
        sigmoid_input = (
            self.similarity_weight * similarity_score -  # Reward high similarity
            self.rank_weight * math.log(article_rank) +   # Penalize high rank (log scale)
            self.sigmoid_bias                             # Bias term
        )
        
        # Sigmoid function: 1 / (1 + e^(-x))
        try:
            sigmoid_output = 1.0 / (1.0 + math.exp(-sigmoid_input))
        except OverflowError:
            # Handle extreme values
            sigmoid_output = 1.0 if sigmoid_input > 0 else 0.0
        
        # Scale by max boost factor
        final_boost = sigmoid_output * self.max_boost_factor
        
        return final_boost
    
    def get_boost_explanation(self, similarity_score: float, article_rank: int, boost_value: float) -> str:
        """Tạo explanation cho boost value để debug"""
        if not self.use_sigmoid_boosting:
            return f"Simple: {self.article_ranking_boost:.3f}/rank{article_rank} = {boost_value:.6f}"
        
        import math
        sigmoid_input = (
            self.similarity_weight * similarity_score -
            self.rank_weight * math.log(article_rank) +
            self.sigmoid_bias
        )
        
        try:
            sigmoid_raw = 1.0 / (1.0 + math.exp(-sigmoid_input))
        except OverflowError:
            sigmoid_raw = 1.0 if sigmoid_input > 0 else 0.0
        
        return (f"Sigmoid: sim({similarity_score:.3f})*{self.similarity_weight} - "
                f"log(rank{article_rank})*{self.rank_weight} + {self.sigmoid_bias} = "
                f"{sigmoid_input:.3f} → σ({sigmoid_raw:.4f}) * {self.max_boost_factor} = {boost_value:.6f}")

def main():
    parser = argparse.ArgumentParser(description='End-to-End Search Pipeline: Text + Image')
    
    # Mode selection
    parser.add_argument('--text-search-only', action='store_true', help='Chỉ chạy cascade text search')
    parser.add_argument('--image-search-only', action='store_true', help='Chỉ chạy image search từ existing CSV files')
    parser.add_argument('--csv-files', nargs='+', help='CSV files for image-search-only mode (no cascade)')
    parser.add_argument('--additional-csv-files', nargs='+', help='Additional CSV files to RRF with cascade results')
    
    # JSON Config Support (NEW)
    parser.add_argument('--json-config', help='JSON config file for model weights and database configuration')
    
    # Text search config
    parser.add_argument('--es-host', default='http://localhost:9200', help='Elasticsearch host')
    parser.add_argument('--expansion-file', default='query_expansion.json', help='Query expansion file')
    parser.add_argument('--blacklist-file', default='celeb_names.txt', help='Celebrity blacklist file')
    parser.add_argument('--use-clean-index', action='store_true', help='Use articles_clean index')
    parser.add_argument('--articles-by-year', default='articles_by_year.json', help='Articles by year file for date filtering')
    parser.add_argument('--config-name', help='Config name for output directory (default: datetime)')
    parser.add_argument('--text-top-k', type=int, default=30, help='Top-k for text search')
    parser.add_argument('--max-queries', type=int, help='Limit queries for testing')
    
    # Image search config
    parser.add_argument('--qdrant-host', default='localhost', help='Qdrant host')
    parser.add_argument('--qdrant-port', type=int, default=6333, help='Qdrant port')
    parser.add_argument('--qdrant-url', help='Full Qdrant URL')
    parser.add_argument('--article-mapping', default='database_article_to_images_v.0.1.json', help='Article mapping file')
    
    # LEGACY: Checkpoint/Dataset selection (deprecated when using JSON config)
    parser.add_argument('--primary-checkpoint', choices=['Initialized', 'Flickr30k', 'OpenEvents_v1'], 
                       default='Initialized', help='Primary checkpoint/dataset for search (LEGACY)')
    parser.add_argument('--enable-h14-laion', action='store_true', default=True, help='Enable H14 Laion as additional model (LEGACY)')
    parser.add_argument('--disable-h14-laion', action='store_true', help='Disable H14 Laion model (LEGACY)')
    
    # LEGACY: Multi-model configuration (deprecated when using JSON config)
    parser.add_argument('--enable-multi-model', action='store_true', default=True, help='Enable multi-model search (LEGACY)')
    parser.add_argument('--disable-multi-model', action='store_true', help='Disable multi-model (LEGACY)')
    
    # LEGACY: Primary checkpoint weights (deprecated when using JSON config)
    parser.add_argument('--primary-query-large-weight', type=float, default=1.0, help='Primary checkpoint Query-Large weight (LEGACY)')
    parser.add_argument('--primary-summary-large-weight', type=float, default=0.0, help='Primary checkpoint Summary-Large weight (LEGACY)')
    parser.add_argument('--primary-concise-large-weight', type=float, default=0.0, help='Primary checkpoint Concise-Large weight (LEGACY)')
    
    # LEGACY: Primary checkpoint weights (Base model) (deprecated when using JSON config)
    parser.add_argument('--primary-query-base-weight', type=float, default=1.0, help='Primary checkpoint Query-Base weight (LEGACY)')
    parser.add_argument('--primary-summary-base-weight', type=float, default=0.0, help='Primary checkpoint Summary-Base weight (LEGACY)')
    parser.add_argument('--primary-concise-base-weight', type=float, default=0.0, help='Primary checkpoint Concise-Base weight (LEGACY)')
    
    # LEGACY: H14 Laion weights (deprecated when using JSON config)
    parser.add_argument('--h14-query-weight', type=float, default=1.0, help='H14 Laion Query weight (LEGACY)')
    parser.add_argument('--h14-summary-weight', type=float, default=0.0, help='H14 Laion Summary weight (LEGACY)')
    parser.add_argument('--h14-concise-weight', type=float, default=0.0, help='H14 Laion Concise weight (LEGACY)')
    
    # LEGACY: Model family weights (deprecated when using JSON config)
    parser.add_argument('--primary-large-family-weight', type=float, default=1.0, help='Primary checkpoint Large family weight (LEGACY)')
    parser.add_argument('--primary-base-family-weight', type=float, default=1.0, help='Primary checkpoint Base family weight (LEGACY)')
    parser.add_argument('--h14-laion-family-weight', type=float, default=1.0, help='H14-Laion family weight (LEGACY)')
    
    # Search parameters
    parser.add_argument('--article-ranking-boost', type=float, default=0.3, help='Article ranking boost factor')
    parser.add_argument('--rrf-k', type=int, default=50, help='RRF parameter k')
    parser.add_argument('--multi-model-rrf-k', type=int, default=50, help='Multi-model RRF parameter k')
    parser.add_argument('--max-articles-per-query', type=int, default=30, help='Max articles per query')
    parser.add_argument('--direct-search-top-k', type=int, default=30, help='Direct search top-k')
    parser.add_argument('--final-top-k', type=int, default=20, help='Final top-k after collection RRF (before multi-model RRF)')
    parser.add_argument('--adaptive-rrf', action='store_true', default=True, help='Use adaptive RRF')
    parser.add_argument('--normal-rrf', action='store_true', help='Use normal RRF')
    
    # Voting vs RRF mode (NEW)
    parser.add_argument('--use-voting', action='store_true', help='Use voting instead of RRF for aggregating search results')
    parser.add_argument('--use-rrf', action='store_true', default=True, help='Use RRF for aggregating search results (default)')
    
    # Sigmoid Boosting parameters
    parser.add_argument('--use-sigmoid-boosting', action='store_true', default=True, help='Use advanced sigmoid boosting')
    parser.add_argument('--disable-sigmoid-boosting', action='store_true', help='Disable sigmoid boosting (use simple)')
    parser.add_argument('--similarity-threshold', type=float, default=0.6, help='Similarity threshold for boosting (optimized)')
    parser.add_argument('--similarity-weight', type=float, default=10.0, help='Similarity weight in sigmoid function (optimized)')
    parser.add_argument('--rank-weight', type=float, default=2.0, help='Article rank weight in sigmoid function (optimized: balanced penalty)')
    parser.add_argument('--sigmoid-bias', type=float, default=0.0, help='Bias term in sigmoid function')
    parser.add_argument('--max-boost-factor', type=float, default=0.5, help='Maximum boost factor to apply')
    
    # Debug
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Private test mode
    parser.add_argument('--private-test', action='store_true', help='Enable private test mode (add Private_ prefix to query collections)')
    
    args = parser.parse_args()
    
    # Validate additional CSV files 
    if args.additional_csv_files and len(args.additional_csv_files) < 1:
        print(" At least 1 additional CSV file is required if specified")
        sys.exit(1)
    
    # Validate JSON config
    if args.json_config and not os.path.exists(args.json_config):
        print(f" JSON config file not found: {args.json_config}")
        sys.exit(1)
    
    # Print configuration mode
    if args.json_config:
        print(f" CONFIGURATION MODE: JSON config from {args.json_config}")
    else:
        print(" CONFIGURATION MODE: Legacy parameter-based")
    
    # Determine RRF mode
    adaptive_rrf = True
    if args.normal_rrf:
        adaptive_rrf = False
    elif args.adaptive_rrf:
        adaptive_rrf = True
    
    # Determine voting vs RRF mode
    use_voting = False
    if args.use_voting:
        use_voting = True
        print(" AGGREGATION MODE: VOTING")
    elif args.use_rrf:
        use_voting = False
        print(" AGGREGATION MODE: RRF")
    
    # Determine sigmoid boosting mode
    use_sigmoid = True
    if args.disable_sigmoid_boosting:
        use_sigmoid = False
    elif args.use_sigmoid_boosting:
        use_sigmoid = True
    
    # Determine multi-model mode (only for legacy mode)
    enable_multi_model = True
    if not args.json_config:
        if args.disable_multi_model:
            enable_multi_model = False
        elif args.enable_multi_model:
            enable_multi_model = True
    
    # Determine H14 Laion mode (only for legacy mode)
    enable_h14_laion = True
    if not args.json_config:
        if args.disable_h14_laion:
            enable_h14_laion = False
        elif args.enable_h14_laion:
            enable_h14_laion = True
    
    try:
        # Initialize professional search pipeline with enhanced entity search
        pipeline = ProfessionalSearchPipeline(
            es_host=args.es_host,
            expansion_file=args.expansion_file,
            blacklist_file=args.blacklist_file,
            use_clean_index=args.use_clean_index,
            articles_by_year_file=args.articles_by_year,  # NEW: Date filtering
            qdrant_host=args.qdrant_host,
            qdrant_port=args.qdrant_port,
            qdrant_url=args.qdrant_url,
            article_mapping_json=args.article_mapping,
            
            # JSON Config Support (NEW)
            json_config_file=args.json_config,
            
            # Legacy: Checkpoint/Dataset selection
            primary_checkpoint=args.primary_checkpoint,
            enable_h14_laion=enable_h14_laion,
            
            # Private test mode
            private_test_mode=args.private_test,
            
            # Legacy: Multi-model configuration
            enable_multi_model=enable_multi_model,
            
            # Legacy: Primary checkpoint weights (Large)
            primary_query_large_weight=args.primary_query_large_weight,
            primary_summary_large_weight=args.primary_summary_large_weight,
            primary_concise_large_weight=args.primary_concise_large_weight,
            
            # Legacy: Primary checkpoint weights (Base)
            primary_query_base_weight=args.primary_query_base_weight,
            primary_summary_base_weight=args.primary_summary_base_weight,
            primary_concise_base_weight=args.primary_concise_base_weight,
            
            # Legacy: H14 Laion weights
            h14_query_weight=args.h14_query_weight,
            h14_summary_weight=args.h14_summary_weight,
            h14_concise_weight=args.h14_concise_weight,
            
            # Legacy: Model family weights
            primary_large_family_weight=args.primary_large_family_weight,
            primary_base_family_weight=args.primary_base_family_weight,
            h14_laion_family_weight=args.h14_laion_family_weight,
            
            article_ranking_boost=args.article_ranking_boost,
            rrf_k=args.rrf_k,
            multi_model_rrf_k=args.multi_model_rrf_k,
            use_voting=use_voting,  # NEW: Voting mode
            use_sigmoid_boosting=use_sigmoid,
            similarity_threshold=args.similarity_threshold,
            similarity_weight=args.similarity_weight,
            rank_weight=args.rank_weight,
            sigmoid_bias=args.sigmoid_bias,
            max_boost_factor=args.max_boost_factor,
            debug=args.debug
        )
        
        # Run appropriate pipeline
        if args.text_search_only:
            # CASCADE text search only
            output_dir = pipeline.create_output_directory(args.config_name)
            pipeline.save_config(output_dir, args, mode="cascade_text_search_only")
            
            csv_path, json_path = pipeline.text_search_pipeline(output_dir, "cascade", args.text_top_k, args.max_queries)
            print(f" CASCADE text search completed!")
            print(f" CSV: {csv_path}")
            print(f" JSON: {json_path}")
            
        elif args.image_search_only:
            # Image search only (từ existing CSV files - NO CASCADE)
            if not args.csv_files:
                print(" Image search only requires --csv-files parameter")
                sys.exit(1)
            
            output_dir = pipeline.create_output_directory(args.config_name)
            pipeline.save_config(output_dir, args, mode="image_search_only")
            
            # Process CSV files
            submission_csv_path = pipeline.rrf_rerank_csvs(args.csv_files, adaptive_rrf, output_dir)
            
            # Image search (multi-model or single-model)
            if enable_multi_model:
                final_results = pipeline.multi_model_image_search_pipeline(
                    submission_csv_path,
                    max_articles_per_query=args.max_articles_per_query,
                    direct_search_top_k=args.direct_search_top_k,
                    final_top_k=args.final_top_k
                )
            else:
                final_results = pipeline.image_search_pipeline(
                    submission_csv_path,
                    max_articles_per_query=args.max_articles_per_query,
                    direct_search_top_k=args.direct_search_top_k,
                    final_top_k=args.final_top_k
                )
            
            # Save track2 results
            filename_suffix = args.config_name if args.config_name else "image_search"
            track2_path = pipeline.save_final_image_results(final_results, output_dir, filename_suffix)
            
            print(f" Image search completed!")
            print(f" Submission: {submission_csv_path}")
            print(f" Track2: {track2_path}")
            
        else:
            # INTEGRATED CASCADE + IMAGE pipeline
            submission_path, stage1_path, track2_path = pipeline.run_integrated_cascade_pipeline(
                config_name=args.config_name,
                text_top_k=args.text_top_k,
                max_queries=args.max_queries,
                additional_csv_files=args.additional_csv_files,
                adaptive_rrf=adaptive_rrf,
                args=args  # Pass args for config saving
            )
            print(f" INTEGRATED CASCADE pipeline completed!")
            print(f" Submission: {submission_path}")
            if stage1_path:
                print(f" Stage 1: {stage1_path}")
            print(f" Track 2: {track2_path}")
        
    except Exception as e:
        print(f" Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 
