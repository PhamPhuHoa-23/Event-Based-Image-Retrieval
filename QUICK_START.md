# Quick Start Guide

## What's Been Done

### 1. Codebase Cleaned and Reorganized
- **101 files moved** to `../ElasticSearchSystem_Archive/`
- **Professional file names** applied
- **16 essential files** kept for core functionality

### 2. File Renames (Professional Names)
```
OLD NAME                                 → NEW NAME
enhanced_e2e_pipeline_with_date_filter.py → search_pipeline.py
retrieval_test_enhanced.py               → entity_search_system.py
upload_and_clean_articles_streaming.py  → elasticsearch_articles_uploader.py
upload_private_queries.py               → elasticsearch_queries_uploader.py
retrieval_test.py                       → (archived - using notebooks)
es_by_ct.ipynb                          → elasticsearch_search_notebook.ipynb
es_ct_upload.ipynb                      → elasticsearch_upload_notebook.ipynb
```

### 3. Enhanced System Architecture
- **Entity Search System**: Weighted entity scoring for better accuracy
- **Professional Pipeline**: Clean, maintainable code structure  
- **Web Interface**: Complete result visualization system
- **Upload Tools**: Professional Elasticsearch data management

## Current Directory Structure

```
ElasticSearchSystem/
├── CORE SEARCH PIPELINE
│   ├── search_pipeline.py              # Main search pipeline
│   ├── entity_search_system.py         # Enhanced entity search
│   └── rrf_rerank.py                   # Ranking system
│
├── WEB APPLICATION
│   ├── app.py                          # Main web interface
│   ├── debug_app.py                    # Debug interface
│   ├── templates/                      # Web templates
│   └── static/                         # Web assets
│
├── ELASTICSEARCH TOOLS
│   ├── elasticsearch_search_notebook.ipynb    # Full-text search
│   ├── elasticsearch_upload_notebook.ipynb    # Data upload
│   ├── elasticsearch_articles_uploader.py     # Articles uploader
│   └── elasticsearch_queries_uploader.py      # Queries uploader
│
├── EXAMPLE & TESTING
│   ├── example_data_generator.py       # Generate test data
│   ├── app_results/                    # Result storage
│   └── csv_app/                        # CSV files
│
└── DATA & CONFIG
    ├── config.json                     # Pipeline config
    ├── query_expansion.json            # Query expansion
    ├── articles_by_year.json           # Date filtering
    ├── database_article_to_images_v.0.1.json # Article mapping
    └── celeb_blacklist.csv             # Celebrity blacklist
```

## Quick Test (5 Minutes)

### 1. Test Web Interface
```bash
python app.py
# Open http://localhost:5000
# Select 'example_search_TIMESTAMP' from dropdown
```

### 2. Test Entity Search
```bash
python entity_search_system.py --mode sample --sample-size 3
```

### 3. Test Pipeline
```bash
python search_pipeline.py --text-search-only --max-queries 5 --debug
```

## Complete Workflow

### Step 1: Upload Data to Elasticsearch
```bash
# Upload articles
jupyter notebook elasticsearch_upload_notebook.ipynb

# Upload queries  
python elasticsearch_queries_uploader.py --input queries.json --index queries
```

### Step 2: Generate Entity Search Results
```bash
python entity_search_system.py --mode search_all --top_k 30 --postfix entity
# Output: submission_entity.csv
```

### Step 3: Generate Full-text Search Results
```bash
jupyter notebook elasticsearch_search_notebook.ipynb
# Follow notebook instructions to generate submission_fulltext.csv
```

### Step 4: Combine with Image Search
```bash
python search_pipeline.py --image-search-only \
  --csv-files submission_entity.csv submission_fulltext.csv \
  --use-voting \
  --config-name "combined_search"
```

### Step 5: View Results
```bash
python app.py
# Open http://localhost:5000
# Select result set from dropdown
```

## File Formats

### Entity Search Output
```csv
query_id,article_id_1,article_id_2,...,article_id_30
Q001,A1234,A5678,A9012,...
```

### Image Search Output  
```csv
query_id,image_id_1,image_id_2,...,image_id_10
Q001,IMG001,IMG002,IMG003,...
```

## Key Features

### Enhanced Entity Search
- **Weighted scoring**: PERSON(4.3), CARDINAL(3.5), ORG(3.8), etc.
- **Optimized weights** from data analysis
- **Private/Public query support**
- **Clean articles index support**

### Professional Pipeline
- **JSON configuration** support
- **Voting and RRF** aggregation modes
- **Multi-model image search**
- **Comprehensive logging**

### Web Interface
- **Result set selection** from app_results/
- **Query browsing** with entity highlighting
- **Image preview** with article links
- **Stage 1 text results** visualization

## Troubleshooting

### Common Issues
1. **"No Elasticsearch connection"**
   - Start Elasticsearch: Check `http://localhost:9200`
   - Verify indices exist

2. **"No Qdrant connection"**  
   - Start Qdrant: Check `http://localhost:6333`
   - Verify collections exist

3. **"Web interface empty"**
   - Run `python example_data_generator.py` first
   - Select result set from dropdown

4. **"Entity search fails"**
   - Check Elasticsearch indices exist
   - Verify query format is correct

### Performance Tips
- Use `--max-queries` for testing small subsets
- Enable `--clean-articles` for cleaner data
- Reduce `--top-k` for faster results
- Use `--debug` for detailed logging

## What's Ready to Use

### Immediately Available
- **Web Interface**: Full functionality with example data
- **Entity Search**: Production-ready with optimized weights  
- **Pipeline System**: Complete end-to-end processing
- **Upload Tools**: Professional data management

### Requires Setup
- **Elasticsearch indices**: Upload your data first
- **Qdrant collections**: Image embeddings needed
- **Real data files**: Replace example data with actual datasets

## Next Steps

1. **Upload your data** using the notebooks and upload tools
2. **Configure entity weights** in entity_search_system.py if needed
3. **Customize web interface** templates and styles
4. **Add custom search logic** as needed

The system is now professional, clean, and ready for production use! 