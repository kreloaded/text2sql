"""
Preprocess Spider dataset into FAISS-ready format.
Converts schema information and question-SQL pairs into embeddings-ready JSON.
"""

import json
import argparse
import os
from pathlib import Path


def load_spider_tables(tables_path):
    """Load Spider tables.json file containing schema information."""
    with open(tables_path, 'r') as f:
        return json.load(f)


def load_spider_dataset(dataset_path):
    """Load Spider train/dev dataset containing question-SQL pairs."""
    with open(dataset_path, 'r') as f:
        return json.load(f)


def format_schema_text(table_info):
    """
    Format table schema into natural language text for embedding.
    
    Args:
        table_info: Dict containing table schema from Spider tables.json
    
    Returns:
        List of dicts with 'text' and 'meta' keys for each table
    """
    db_id = table_info['db_id']
    table_names = table_info['table_names_original']
    column_names = table_info['column_names_original']
    column_types = table_info['column_types']
    primary_keys = table_info['primary_keys']
    foreign_keys = table_info['foreign_keys']
    
    schema_entries = []
    
    # Group columns by table
    tables_with_columns = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:  # Skip the special "*" column
            continue
        if table_idx not in tables_with_columns:
            tables_with_columns[table_idx] = []
        tables_with_columns[table_idx].append({
            'name': col_name,
            'type': column_types[col_idx],
            'is_primary': col_idx in primary_keys
        })
    
    # Create schema text for each table
    for table_idx, table_name in enumerate(table_names):
        if table_idx not in tables_with_columns:
            continue
            
        columns = tables_with_columns[table_idx]
        column_descriptions = []
        
        for col in columns:
            desc = f"{col['name']} ({col['type']})"
            if col['is_primary']:
                desc += " [PRIMARY KEY]"
            column_descriptions.append(desc)
        
        # Find foreign keys for this table
        fk_descriptions = []
        for fk_pair in foreign_keys:
            col_idx_from, col_idx_to = fk_pair
            table_from = column_names[col_idx_from][0]
            col_from = column_names[col_idx_from][1]
            table_to = column_names[col_idx_to][0]
            col_to = column_names[col_idx_to][1]
            
            if table_from == table_idx:
                fk_descriptions.append(
                    f"{col_from} -> {table_names[table_to]}.{col_to}"
                )
        
        fk_text = ", ".join(fk_descriptions) if fk_descriptions else "none"
        
        schema_text = (
            f"Database: {db_id}, Table: {table_name}, "
            f"Columns: {', '.join(column_descriptions)}, "
            f"Foreign Keys: {fk_text}"
        )
        
        schema_entries.append({
            'text': schema_text,
            'meta': {
                'type': 'schema',
                'db_id': db_id,
                'table_name': table_name,
                'columns': [col['name'] for col in columns]
            }
        })
    
    return schema_entries


def format_example_text(example):
    """
    Format question-SQL pair into natural language text for embedding.
    
    Args:
        example: Dict containing question and SQL from Spider dataset
    
    Returns:
        Dict with 'text' and 'meta' keys
    """
    question = example['question']
    sql = example['query']
    db_id = example['db_id']
    
    # Combine question and SQL for embedding
    example_text = f"Question: {question}\nSQL: {sql}"
    
    return {
        'text': example_text,
        'meta': {
            'type': 'example',
            'db_id': db_id,
            'question': question,
            'sql': sql
        }
    }


def preprocess_spider(tables_path, dataset_path, output_path, include_examples=True):
    """
    Main preprocessing function.
    
    Args:
        tables_path: Path to Spider tables.json
        dataset_path: Path to Spider train.json or dev.json
        output_path: Path to save preprocessed JSON
        include_examples: Whether to include example question-SQL pairs
    """
    print("Loading Spider dataset...")
    tables = load_spider_tables(tables_path)
    dataset = load_spider_dataset(dataset_path) if include_examples else []
    
    all_entries = []
    
    # Process schemas
    print(f"Processing {len(tables)} database schemas...")
    for table_info in tables:
        schema_entries = format_schema_text(table_info)
        all_entries.extend(schema_entries)
    
    print(f"Generated {len(all_entries)} schema entries")
    
    # Process examples
    if include_examples:
        print(f"Processing {len(dataset)} question-SQL examples...")
        for example in dataset:
            example_entry = format_example_text(example)
            all_entries.append(example_entry)
        
        print(f"Total entries: {len(all_entries)} (schemas + examples)")
    
    # Save to output
    print(f"Saving to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(all_entries, f, indent=2)
    
    print(f"✓ Preprocessing complete! Saved {len(all_entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Spider dataset for FAISS ingestion."
    )
    parser.add_argument(
        '--tables',
        required=True,
        help='Path to Spider tables.json file'
    )
    parser.add_argument(
        '--dataset',
        required=True,
        help='Path to Spider train.json or dev.json file'
    )
    parser.add_argument(
        '--output',
        default='spider_processed.json',
        help='Path to save preprocessed JSON file'
    )
    parser.add_argument(
        '--schemas-only',
        action='store_true',
        help='Only process schemas, skip example question-SQL pairs'
    )
    
    args = parser.parse_args()
    
    preprocess_spider(
        args.tables,
        args.dataset,
        args.output,
        include_examples=not args.schemas_only
    )


if __name__ == "__main__":
    main()
