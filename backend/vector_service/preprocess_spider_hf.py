"""
Load Spider dataset from HuggingFace and preprocess for FAISS ingestion.
This script downloads the dataset and converts it to the required format.
"""

from datasets import load_dataset
import json
import argparse


def load_spider_from_hf():
    """Load Spider dataset from HuggingFace."""
    print("Loading Spider dataset from HuggingFace...")
    raw_datasets = load_dataset("xlangai/spider")
    return raw_datasets


def extract_schema_from_dataset(dataset):
    """Extract unique database schemas from the dataset."""
    schemas_dict = {}

    for example in dataset:
        db_id = example["db_id"]
        if db_id not in schemas_dict:
            # Extract schema information from the example
            schemas_dict[db_id] = {
                "db_id": db_id,
                "table_names": example.get("db_table_names", []),
                "column_names": example.get("db_column_names", {}),
                "column_types": example.get("db_column_types", []),
                "primary_keys": example.get("db_primary_keys", []),
                "foreign_keys": example.get("db_foreign_keys", []),
            }

    return list(schemas_dict.values())


def format_schema_entry(db_info):
    """Format database schema into FAISS-ready format with detailed structure."""
    db_id = db_info["db_id"]
    table_names = db_info.get("table_names", [])
    column_names = db_info.get("column_names", {})
    column_types = db_info.get("column_types", [])

    # Build detailed schema text
    schema_lines = [f"Database: {db_id}"]

    if table_names and column_names:
        schema_lines.append("\nTables and Columns:")

        # Group columns by table
        tables_dict = {}

        # column_names is typically {'table_id': [-1, 0, 0, 1, 1, ...], 'column_name': ['*', 'id', 'name', ...]}
        if isinstance(column_names, dict):
            table_ids = column_names.get("table_id", [])
            col_names = column_names.get("column_name", [])

            for i, (table_id, col_name) in enumerate(zip(table_ids, col_names)):
                if table_id == -1:  # Skip * column
                    continue

                if table_id not in tables_dict:
                    table_name = (
                        table_names[table_id]
                        if table_id < len(table_names)
                        else f"table_{table_id}"
                    )
                    tables_dict[table_id] = {"name": table_name, "columns": []}

                col_type = column_types[i] if i < len(column_types) else "unknown"
                tables_dict[table_id]["columns"].append(f"{col_name} ({col_type})")

        # Format tables with their columns
        for table_id in sorted(tables_dict.keys()):
            table_info = tables_dict[table_id]
            schema_lines.append(
                f"  - {table_info['name']}: {', '.join(table_info['columns'])}"
            )

    schema_text = "\n".join(schema_lines)

    return {
        "text": schema_text,
        "meta": {
            "type": "schema",
            "db_id": db_id,
            "tables": table_names,
            "schema": schema_text,  # Store full schema in metadata too
        },
    }


def format_example_entry(example):
    """Format question-SQL pair into FAISS-ready format."""
    question = example["question"]
    sql = example["query"]
    db_id = example["db_id"]

    example_text = f"Question: {question}\nSQL: {sql}"

    return {
        "text": example_text,
        "meta": {"type": "example", "db_id": db_id, "question": question, "sql": sql},
    }


def preprocess_hf_spider(
    output_path, split="train", include_schemas=True, include_examples=True
):
    """
    Load Spider from HuggingFace and preprocess for FAISS.

    Args:
        output_path: Path to save preprocessed JSON
        split: Dataset split ('train' or 'validation')
        include_schemas: Whether to include schema entries
        include_examples: Whether to include example question-SQL pairs
    """
    raw_datasets = load_spider_from_hf()
    dataset = raw_datasets[split]

    print(f"Processing {len(dataset)} examples from {split} split...")

    all_entries = []

    # Extract and process schemas
    if include_schemas:
        print("Extracting database schemas...")
        schemas = extract_schema_from_dataset(dataset)
        print(f"Found {len(schemas)} unique databases")

        for schema in schemas:
            schema_entry = format_schema_entry(schema)
            all_entries.append(schema_entry)

        print(f"Generated {len(schemas)} schema entries")

    # Process examples
    if include_examples:
        print("Processing question-SQL examples...")
        for example in dataset:
            example_entry = format_example_entry(example)
            all_entries.append(example_entry)

        print(f"Generated {len(dataset)} example entries")

    print(f"Total entries: {len(all_entries)}")

    # Save to output
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(
        f"✓ Preprocessing complete! Saved {len(all_entries)} entries to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load Spider from HuggingFace and preprocess for FAISS ingestion."
    )
    parser.add_argument(
        "--output",
        default="spider_hf_processed.json",
        help="Path to save preprocessed JSON file",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to process (train or validation)",
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

    args = parser.parse_args()

    include_schemas = not args.examples_only
    include_examples = not args.schemas_only

    preprocess_hf_spider(args.output, args.split, include_schemas, include_examples)


if __name__ == "__main__":
    main()
