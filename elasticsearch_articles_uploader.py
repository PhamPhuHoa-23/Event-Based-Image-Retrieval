#!/usr/bin/env python3
"""
Script để upload file articles.json mới và clean duplicate entities
Streaming version for large files (3GB+)
"""

import json
import requests
from typing import Dict, List, Any
import time
from datetime import datetime
import os
import ijson

class ArticleUploadAndCleaner:
    def __init__(self, es_host="http://localhost:9200"):
        self.es_host = es_host
        self.articles_index = "articles"
        self.articles_clean_index = "articles_clean"
        self.articles_backup_index = "articles_backup"
    
    def backup_existing_index(self):
        """Backup existing index before uploading new data"""
        print(" Backing up existing articles index...")
        
        try:
            # Check if articles index exists
            check_response = requests.head(f"{self.es_host}/{self.articles_index}")
            if check_response.status_code == 404:
                print(" No existing articles index to backup")
                return True
            
            # Delete old backup if exists
            delete_response = requests.delete(f"{self.es_host}/{self.articles_backup_index}")
            if delete_response.status_code in [200, 404]:
                print(f" Cleaned old backup index")
            
            # Reindex articles to backup
            reindex_body = {
                "source": {"index": self.articles_index},
                "dest": {"index": self.articles_backup_index}
            }
            
            reindex_response = requests.post(
                f"{self.es_host}/_reindex",
                headers={"Content-Type": "application/json"},
                json=reindex_body
            )
            
            if reindex_response.status_code == 200:
                result = reindex_response.json()
                total = result.get("total", 0)
                print(f" Backed up {total:,} articles to {self.articles_backup_index}")
                return True
            else:
                print(f" Backup failed: {reindex_response.status_code}")
                return False
                
        except Exception as e:
            print(f" Backup error: {e}")
            return False
    
    def delete_and_recreate_index(self):
        """Delete existing articles index and recreate with proper mapping"""
        print(" Recreating articles index...")
        
        try:
            # Delete existing index
            delete_response = requests.delete(f"{self.es_host}/{self.articles_index}")
            if delete_response.status_code in [200, 404]:
                print(f" Deleted existing {self.articles_index}")
            
            # Create new index with mapping
            mapping = {
                "mappings": {
                    "properties": {
                        "article_id": {
                            "type": "keyword"
                        },
                        "entities": {
                            "type": "nested",
                            "properties": {
                                "text": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword"
                                        }
                                    }
                                },
                                "label": {
                                    "type": "keyword"
                                }
                            }
                        }
                    }
                }
            }
            
            create_response = requests.put(
                f"{self.es_host}/{self.articles_index}",
                headers={"Content-Type": "application/json"},
                json=mapping
            )
            
            if create_response.status_code == 200:
                print(f" Created new {self.articles_index} with proper mapping")
                return True
            else:
                print(f" Failed to create index: {create_response.status_code}")
                return False
                
        except Exception as e:
            print(f" Index creation error: {e}")
            return False
    
    def upload_articles_from_file(self, file_path, batch_size=500):
        """Upload articles from JSON file using streaming approach"""
        print(f" Uploading articles from {file_path} (streaming mode)...")
        
        if not os.path.exists(file_path):
            print(f" File not found: {file_path}")
            return False
        
        try:
            # Stream parse the JSON file
            total_uploaded = 0
            batch_data = []
            article_count = 0
            
            print(f" Starting streaming upload (batch size: {batch_size})...")
            
            with open(file_path, 'rb') as f:
                # Parse JSON object with article_id as keys
                for article_id, entities in ijson.kvitems(f, ''):
                    article_count += 1
                    
                    # Create article object
                    article = {
                        "article_id": article_id,
                        "entities": entities
                    }
                    
                    # Prepare bulk index operation
                    batch_data.append({
                        "index": {
                            "_index": self.articles_index,
                            "_id": article_id
                        }
                    })
                    batch_data.append(article)
                    
                    # Upload batch when size reached
                    if len(batch_data) >= batch_size * 2:  # *2 because each article has 2 elements
                        success = self._bulk_upload(batch_data)
                        if success:
                            total_uploaded += batch_size
                            print(f"⏳ Uploaded: {total_uploaded:,} articles ({article_count:,} processed)")
                        else:
                            print(f" Batch upload failed at {total_uploaded}")
                            return False
                        
                        batch_data = []
            
            # Upload remaining batch
            if batch_data:
                success = self._bulk_upload(batch_data)
                if success:
                    remaining = len(batch_data) // 2
                    total_uploaded += remaining
                    print(f"⏳ Uploaded final batch: {remaining} articles")
            
            print(f" Upload completed: {total_uploaded:,} articles")
            return True
            
        except Exception as e:
            print(f" Upload error: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            return False
    
    def _bulk_upload(self, batch_data):
        """Bulk upload batch data"""
        try:
            bulk_body = "\n".join([json.dumps(item) for item in batch_data]) + "\n"
            
            response = requests.post(
                f"{self.es_host}/_bulk",
                headers={"Content-Type": "application/x-ndjson"},
                data=bulk_body
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("errors", False):
                    print(f" Some upload errors occurred:")
                    # Show first few errors for debugging
                    for item in result.get("items", [])[:3]:
                        if "index" in item and "error" in item["index"]:
                            error = item["index"]["error"]
                            print(f"    Error: {error.get('type', 'unknown')} - {error.get('reason', 'no reason')}")
                    return False
                return True
            else:
                print(f" Bulk upload error: {response.status_code}")
                print(f"   Response: {response.text[:500]}")
                return False
                
        except Exception as e:
            print(f" Bulk upload exception: {e}")
            return False
    
    def refresh_index(self):
        """Refresh index to make documents searchable"""
        try:
            response = requests.post(f"{self.es_host}/{self.articles_index}/_refresh")
            if response.status_code == 200:
                print(f" Refreshed {self.articles_index}")
                return True
            else:
                print(f" Refresh failed: {response.status_code}")
                return False
        except Exception as e:
            print(f" Refresh error: {e}")
            return False
    
    def verify_upload(self):
        """Verify uploaded data"""
        print(" Verifying uploaded data...")
        
        try:
            # Count documents
            count_response = requests.get(f"{self.es_host}/{self.articles_index}/_count")
            if count_response.status_code == 200:
                count = count_response.json()["count"]
                print(f" Total articles in index: {count:,}")
            
            # Sample check
            sample_response = requests.post(
                f"{self.es_host}/{self.articles_index}/_search",
                headers={"Content-Type": "application/json"},
                json={"size": 3, "query": {"match_all": {}}}
            )
            
            if sample_response.status_code == 200:
                docs = sample_response.json()["hits"]["hits"]
                print(f" Sample check:")
                
                total_entities = 0
                total_unique = 0
                
                for doc in docs:
                    article_id = doc["_source"]["article_id"]
                    entities = doc["_source"].get("entities", [])
                    
                    # Check duplicates
                    entity_keys = [f"{e.get('text', '').lower()}|{e.get('label', '')}" for e in entities]
                    unique_keys = set(entity_keys)
                    
                    total_entities += len(entities)
                    total_unique += len(unique_keys)
                    
                    duplicate_count = len(entities) - len(unique_keys)
                    print(f"  Article {article_id}: {len(entities)} entities, {duplicate_count} duplicates")
                
                avg_duplicates = (total_entities - total_unique) / len(docs) if docs else 0
                print(f" Average duplicates per article: {avg_duplicates:.1f}")
                
                return True
            else:
                print(f" Sample check failed: {sample_response.status_code}")
                return False
                
        except Exception as e:
            print(f" Verification error: {e}")
            return False
    
    def clean_duplicates(self):
        """Clean duplicate entities using the existing script"""
        print(" Starting duplicate entity cleaning...")
        
        try:
            from fix_duplicate_entities_index import ElasticsearchEntityDeduplicator
            
            deduplicator = ElasticsearchEntityDeduplicator(self.es_host)
            
            # Override index names to use our current setup
            deduplicator.old_index = self.articles_index
            deduplicator.new_index = self.articles_clean_index
            
            # Create clean index
            if not deduplicator.create_clean_index():
                print(" Failed to create clean index")
                return False
            
            # Process articles
            if not deduplicator.process_articles_batch(batch_size=200):
                print(" Failed to clean duplicates")
                return False
            
            # Verify clean index
            deduplicator.verify_clean_index()
            
            print(" Duplicate cleaning completed!")
            return True
            
        except Exception as e:
            print(f" Cleaning error: {e}")
            return False

def main():
    print(" ARTICLE UPLOAD AND CLEANER (STREAMING)")
    print("=" * 55)
    
    # Auto-detect articles.json file
    file_path = "articles.json"
    
    if not os.path.exists(file_path):
        print(f" File not found: {file_path}")
        print(" Make sure articles.json is in the current directory")
        return
    
    print(f" Found articles file: {file_path}")
    
    # Get file size for info
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
    print(f" File size: {file_size:.1f} MB")
    
    # Show plan and auto-proceed
    print(f"\n PLAN:")
    print(f"1. Backup existing articles index")
    print(f"2. Upload new articles from: {file_path} (streaming)")
    print(f"3. Clean duplicate entities")
    print(f"4. Verify results")
    print(f"\n Starting process automatically...")
    
    # Initialize uploader
    uploader = ArticleUploadAndCleaner()
    
    # Step 1: Backup
    print(f"\n1⃣ Backing up existing data...")
    if not uploader.backup_existing_index():
        print(" Backup failed - stopping")
        return
    
    # Step 2: Recreate and upload
    print(f"\n2⃣ Uploading new articles...")
    if not uploader.delete_and_recreate_index():
        print(" Index recreation failed - stopping")
        return
    
    if not uploader.upload_articles_from_file(file_path, batch_size=500):  # Smaller batches for stability
        print(" Upload failed - stopping")
        return
    
    # Refresh index
    uploader.refresh_index()
    
    # Step 3: Verify upload
    print(f"\n3⃣ Verifying upload...")
    if not uploader.verify_upload():
        print(" Verification failed - stopping")
        return
    
    # Step 4: Clean duplicates
    print(f"\n4⃣ Cleaning duplicates...")
    if not uploader.clean_duplicates():
        print(" Cleaning failed")
        return
    
    print(f"\n ALL DONE!")
    print(f" New articles uploaded to 'articles'")
    print(f" Clean articles available in 'articles_clean'")
    print(f" Backup available in 'articles_backup'")
    print(f"\n Use --use_clean_index flag for best results!")

if __name__ == "__main__":
    main() 