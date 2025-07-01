#!/usr/bin/env python3
"""
Upload private_entities.json to Elasticsearch as 'private_queries' index
"""
import json
import requests
from datetime import datetime
from system_test import SafeElasticsearchEntityManager

class PrivateQueriesUploader:
    def __init__(self, es_host="http://localhost:9200"):
        self.es_host = es_host
        self.private_queries_index = "private_queries"
        self.es_manager = SafeElasticsearchEntityManager(es_host)

    def create_private_queries_index(self):
        """Tạo index private_queries với mapping giống queries"""
        
        private_queries_mapping = {
            "mappings": {
                "properties": {
                    "query_id": {"type": "keyword"},
                    "query_text": {"type": "text", "analyzer": "standard"},
                    "entities": {
                        "type": "nested",
                        "properties": {
                            "text": {
                                "type": "text",
                                "analyzer": "standard", 
                                "fields": {"keyword": {"type": "keyword"}}
                            },
                            "label": {"type": "keyword"},
                            "label_description": {"type": "text"},
                            "start_char": {"type": "integer"},
                            "end_char": {"type": "integer"},
                            "confidence": {"type": "float"}
                        }
                    },
                    "entity_count": {"type": "integer"},
                    "processed_at": {"type": "date"},
                    "data_type": {"type": "keyword"},  # "private"
                    "created_at": {"type": "date"},
                    "updated_at": {"type": "date"}
                }
            }
        }
        
        # Check if index exists
        if self.es_manager.index_exists(self.private_queries_index):
            count = self.es_manager.get_index_doc_count(self.private_queries_index)
            print(f"⏩ Index '{self.private_queries_index}' đã tồn tại với {count:,} documents")
            return True
        
        # Create new index
        response = requests.put(
            f"{self.es_host}/{self.private_queries_index}",
            headers={"Content-Type": "application/json"},
            json=private_queries_mapping
        )
        
        if response.status_code in [200, 201]:
            print(f"✅ Đã tạo index mới: {self.private_queries_index}")
            return True
        else:
            print(f"❌ Lỗi tạo index {self.private_queries_index}: {response.text}")
            return False

    def upload_private_queries(self, file_path="private_entities.json", batch_size=100):
        """Upload private queries từ JSON file"""
        
        print(f"📤 Uploading private queries from {file_path}...")
        
        try:
            # Load data
            with open(file_path, 'r', encoding='utf-8') as f:
                private_data = json.load(f)
            
            print(f"📊 Loaded {len(private_data):,} private queries")
            
            # Prepare data for upload
            current_time = datetime.now().isoformat()
            total_queries = len(private_data)
            total_batches = (total_queries + batch_size - 1) // batch_size
            
            print(f"📦 Uploading in {total_batches} batches of {batch_size}...")
            
            uploaded = 0
            for batch_num in range(total_batches):
                start_idx = batch_num * batch_size
                end_idx = min((batch_num + 1) * batch_size, total_queries)
                
                # Get batch data
                batch_keys = list(private_data.keys())[start_idx:end_idx]
                
                print(f"📦 Batch {batch_num + 1}/{total_batches}: Processing {len(batch_keys)} queries...", end=" ")
                
                # Create bulk data
                bulk_data = []
                for query_id in batch_keys:
                    query_info = private_data[query_id]
                    
                    doc = {
                        "query_id": query_id,
                        "query_text": query_info.get("query_text", ""),
                        "entities": query_info.get("entities", []),
                        "entity_count": query_info.get("entity_count", 0),
                        "processed_at": query_info.get("processed_at"),
                        "data_type": "private",
                        "created_at": current_time,
                        "updated_at": current_time
                    }
                    
                    # Bulk format
                    bulk_data.append(json.dumps({
                        "index": {"_index": self.private_queries_index}
                    }))
                    bulk_data.append(json.dumps(doc))
                
                # Send bulk request
                bulk_body = "\n".join(bulk_data) + "\n"
                response = requests.post(
                    f"{self.es_host}/_bulk",
                    headers={"Content-Type": "application/x-ndjson"},
                    data=bulk_body,
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    errors = sum(1 for item in result.get('items', []) if 'error' in item.get('index', {}))
                    success = len(result.get('items', [])) - errors
                    uploaded += success
                    print(f"✅ Success: {success}, Errors: {errors}")
                    
                    if errors > 0:
                        print(f"⚠️ Some errors in batch {batch_num + 1}")
                else:
                    print(f"❌ Bulk request failed: {response.status_code}")
                    print(response.text[:200])
            
            print(f"\n🎉 Upload completed!")
            print(f"📊 Total uploaded: {uploaded:,}/{total_queries:,} queries")
            
            # Verify upload
            final_count = self.es_manager.get_index_doc_count(self.private_queries_index)
            print(f"📈 Final count in index: {final_count:,}")
            
        except Exception as e:
            print(f"❌ Error uploading: {e}")

    def verify_private_queries(self):
        """Verify uploaded private queries"""
        print("🔍 Verifying private queries...")
        
        try:
            # Get sample
            response = requests.post(
                f"{self.es_host}/{self.private_queries_index}/_search",
                headers={"Content-Type": "application/json"},
                json={"size": 3, "query": {"match_all": {}}}
            )
            
            if response.status_code == 200:
                result = response.json()
                hits = result["hits"]["hits"]
                total = result["hits"]["total"]["value"]
                
                print(f"✅ Total documents: {total:,}")
                print(f"📝 Sample documents:")
                
                for i, hit in enumerate(hits, 1):
                    doc = hit["_source"]
                    print(f"  {i}. Query ID: {doc['query_id']}")
                    print(f"     Text: {doc['query_text'][:80]}...")
                    print(f"     Entities: {doc['entity_count']}")
                    print(f"     Data type: {doc['data_type']}")
                    print()
            else:
                print(f"❌ Search failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Verification error: {e}")

def main():
    print("🚀 PRIVATE QUERIES UPLOADER")
    print("=" * 50)
    
    uploader = PrivateQueriesUploader()
    
    # Step 1: Create index
    print("\n1️⃣ Creating private_queries index...")
    if not uploader.create_private_queries_index():
        print("❌ Failed to create index")
        return
    
    # Step 2: Upload data
    print("\n2️⃣ Uploading private queries...")
    uploader.upload_private_queries()
    
    # Step 3: Verify
    print("\n3️⃣ Verifying upload...")
    uploader.verify_private_queries()
    
    print("\n🎉 PRIVATE QUERIES UPLOAD COMPLETE!")

if __name__ == "__main__":
    main() 