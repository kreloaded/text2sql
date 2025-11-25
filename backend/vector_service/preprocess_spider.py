"""
Preprocess Spider dataset from local files for FAISS ingestion.
This script processes tables.json and training files to create embeddings-ready data.
"""

import json
import argparse
import os


def load_tables(tables_path):
    """Load database schemas from tables.json"""
    print(f"\n{'='*60}")
    print(f"Loading database schemas from {tables_path}...")
    print(f"{'='*60}")

    with open(tables_path, "r", encoding="utf-8") as f:
        tables = json.load(f)

    print(f"✓ Loaded {len(tables)} database schemas")
    return tables


def load_training_examples(train_spider_path, train_others_path=None):
    """Load training examples from JSON files"""
    print(f"\n{'='*60}")
    print("Loading training examples...")
    print(f"{'='*60}")

    examples = []

    # Load train_spider.json
    if os.path.exists(train_spider_path):
        with open(train_spider_path, "r", encoding="utf-8") as f:
            train_spider = json.load(f)
        examples.extend(train_spider)
        print(f"✓ Loaded {len(train_spider)} examples from train_spider.json")

    # Load train_others.json if provided
    if train_others_path and os.path.exists(train_others_path):
        with open(train_others_path, "r", encoding="utf-8") as f:
            train_others = json.load(f)
        examples.extend(train_others)
        print(f"✓ Loaded {len(train_others)} examples from train_others.json")

    print(f"✓ Total training examples: {len(examples)}")
    return examples


def format_schema_for_embedding(schema):
    """
    Format database schema into natural language text for embedding.
    This creates a rich, detailed representation including:
    - Database name
    - All tables with their columns
    - Column types
    - Primary keys
    - Foreign key relationships
    """
    db_id = schema["db_id"]
    table_names = schema["table_names_original"]
    column_names = schema["column_names_original"]
    column_types = schema["column_types"]
    primary_keys = schema["primary_keys"]
    foreign_keys = schema["foreign_keys"]

    # Group columns by table
    tables_with_columns = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:  # Skip the special "*" column
            continue

        if table_idx not in tables_with_columns:
            tables_with_columns[table_idx] = {
                "name": table_names[table_idx],
                "columns": [],
            }

        col_type = column_types[col_idx]
        is_primary = col_idx in primary_keys

        tables_with_columns[table_idx]["columns"].append(
            {
                "name": col_name,
                "type": col_type,
                "is_primary": is_primary,
                "col_idx": col_idx,
            }
        )

    # Build schema text for each table
    schema_entries = []

    for table_idx in sorted(tables_with_columns.keys()):
        table_info = tables_with_columns[table_idx]
        table_name = table_info["name"]
        columns = table_info["columns"]

        # Format column descriptions
        column_descriptions = []
        for col in columns:
            desc = f"{col['name']} ({col['type']})"
            if col["is_primary"]:
                desc += " [PRIMARY KEY]"
            column_descriptions.append(desc)

        # Find foreign keys for this table
        fk_descriptions = []
        for fk_pair in foreign_keys:
            col_idx_from, col_idx_to = fk_pair
            table_from = column_names[col_idx_from][0]

            if table_from == table_idx:
                col_from = column_names[col_idx_from][1]
                table_to = column_names[col_idx_to][0]
                col_to = column_names[col_idx_to][1]
                table_to_name = table_names[table_to]

                fk_descriptions.append(f"{col_from} -> {table_to_name}.{col_to}")

        fk_text = (
            f" | Foreign Keys: {', '.join(fk_descriptions)}" if fk_descriptions else ""
        )

        # Create the schema text
        schema_text = (
            f"Database: {db_id}\n"
            f"Table: {table_name}\n"
            f"Columns: {', '.join(column_descriptions)}{fk_text}"
        )

        schema_entries.append(
            {
                "text": schema_text,
                "meta": {
                    "type": "schema",
                    "db_id": db_id,
                    "table_name": table_name,
                    "columns": [col["name"] for col in columns],
                    "full_schema": schema,  # Store full schema for reference
                },
            }
        )

    return schema_entries


def format_example_for_embedding(example):
    """
    Format question-SQL pair into natural language text for embedding.
    This creates entries that can be used to find similar examples.
    """
    question = example["question"]
    sql = example["query"]
    db_id = example["db_id"]

    # Create a rich text representation
    example_text = f"Database: {db_id}\n" f"Question: {question}\n" f"SQL: {sql}"

    return {
        "text": example_text,
        "meta": {
            "type": "example",
            "db_id": db_id,
            "question": question,
            "sql": sql,
            "query_toks": example.get("query_toks", []),
            "sql_parsed": example.get("sql", {}),
        },
    }


def preprocess_spider(
    tables_path,
    train_spider_path,
    train_others_path,
    output_path,
    database_dir=None,
    include_schemas=True,
    include_examples=True,
):
    """
    Main preprocessing function.

    What gets stored in FAISS:
    1. Schema entries: One entry per table with detailed column info, types, keys
    2. Example entries: Question-SQL pairs for finding similar examples

    This allows:
    - Finding relevant schemas for a new question
    - Finding similar example queries
    - Retrieving complete schema information with relationships
    """
    all_entries = []

    # Process database schemas
    if include_schemas:
        tables = load_tables(tables_path)
        print(f"\n{'='*60}")
        print(f"Processing {len(tables)} database schemas...")
        print(f"{'='*60}")

        for idx, schema in enumerate(tables):
            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(tables)} schemas...")

            schema_entries = format_schema_for_embedding(schema)
            all_entries.extend(schema_entries)

        print(
            f"✓ Generated {len([e for e in all_entries if e['meta']['type'] == 'schema'])} schema entries"
        )

    # Process training examples
    if include_examples:
        examples = load_training_examples(train_spider_path, train_others_path)
        print(f"\n{'='*60}")
        print(f"Processing {len(examples)} training examples...")
        print(f"{'='*60}")

        for idx, example in enumerate(examples):
            if (idx + 1) % 500 == 0:
                print(f"  Processed {idx + 1}/{len(examples)} examples...")

            example_entry = format_example_for_embedding(example)
            all_entries.append(example_entry)

        print(
            f"✓ Generated {len([e for e in all_entries if e['meta']['type'] == 'example'])} example entries"
        )

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total entries: {len(all_entries)}")

    if include_schemas:
        schema_count = sum(1 for e in all_entries if e["meta"]["type"] == "schema")
        unique_dbs = len(
            set(
                e["meta"]["db_id"] for e in all_entries if e["meta"]["type"] == "schema"
            )
        )
        print(f"  - Schema entries: {schema_count} (from {unique_dbs} databases)")

    if include_examples:
        example_count = sum(1 for e in all_entries if e["meta"]["type"] == "example")
        example_dbs = len(
            set(
                e["meta"]["db_id"]
                for e in all_entries
                if e["meta"]["type"] == "example"
            )
        )
        print(f"  - Example entries: {example_count} (from {example_dbs} databases)")

    print(f"{'='*60}")

    # Create output directory if needed
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"\n✓ Created output directory: {output_dir}")

    # Save to output
    print(f"\nSaving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("✓ PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"✓ Saved {len(all_entries)} entries to {output_path}")
    print(f"\nNext step: Run the ingestion script")
    print(f"  python ingest.py --input {output_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Spider dataset from local files for FAISS ingestion."
    )
    parser.add_argument(
        "--spider-dir",
        default="input/spider_data",
        help="Path to Spider dataset directory containing tables.json and training files",
    )
    parser.add_argument(
        "--database-dir",
        help="Path to Spider database folder (default: spider_dir/database)",
    )
    parser.add_argument(
        "--output",
        default="output/spider_processed.json",
        help="Path to save preprocessed JSON file",
    )
    parser.add_argument(
        "--schemas-only",
        action="store_true",
        help="Only process schemas, skip example question-SQL pairs",
    )
    parser.add_argument(
        "--examples-only",
        action="store_true",
        help="Only process examples, skip schemas",
    )
    parser.add_argument(
        "--no-train-others",
        action="store_true",
        help="Skip train_others.json and only use train_spider.json",
    )

    args = parser.parse_args()

    # Build file paths
    spider_dir = args.spider_dir
    tables_path = os.path.join(spider_dir, "tables.json")
    train_spider_path = os.path.join(spider_dir, "train_spider.json")
    train_others_path = (
        None if args.no_train_others else os.path.join(spider_dir, "train_others.json")
    )
    database_dir = args.database_dir or os.path.join(spider_dir, "database")

    # Validate files exist
    if not os.path.exists(tables_path):
        print(f"Error: {tables_path} not found!")
        return

    if not os.path.exists(train_spider_path):
        print(f"Error: {train_spider_path} not found!")
        return

    include_schemas = not args.examples_only
    include_examples = not args.schemas_only

    print("\n" + "=" * 60)
    print("SPIDER DATASET PREPROCESSING FOR FAISS")
    print("=" * 60)
    print(f"Spider directory: {spider_dir}")
    print(f"Database directory: {database_dir}")
    print(f"Output file: {args.output}")
    print(f"Include schemas: {include_schemas}")
    print(f"Include examples: {include_examples}")
    print(f"Include train_others: {not args.no_train_others}")

    preprocess_spider(
        tables_path,
        train_spider_path,
        train_others_path,
        args.output,
        database_dir,
        include_schemas,
        include_examples,
    )


if __name__ == "__main__":
    main()
