"""
Process official Spider dataset (from yale-lily.github.io/spider)
This includes full schema information from tables.json
"""

import json
import argparse
import os


def load_schemas(tables_path):
    """Load database schemas from tables.json"""
    print(f"Loading schemas from {tables_path}...")
    with open(tables_path, "r", encoding="utf-8") as f:
        schemas = json.load(f)
    print(f"Loaded {len(schemas)} database schemas")
    return schemas


def format_schema_text(schema):
    """Format schema into detailed text description"""
    db_id = schema["db_id"]
    table_names = schema.get("table_names_original", schema.get("table_names", []))
    column_names = schema.get("column_names_original", schema.get("column_names", []))
    column_types = schema.get("column_types", [])

    lines = [f"Database: {db_id}"]
    lines.append("\nTables and Columns:")

    # Group columns by table
    tables_dict = {}
    for i, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:  # Skip * column
            continue

        if table_idx not in tables_dict:
            table_name = (
                table_names[table_idx]
                if table_idx < len(table_names)
                else f"table_{table_idx}"
            )
            tables_dict[table_idx] = {"name": table_name, "columns": []}

        col_type = column_types[i] if i < len(column_types) else "unknown"
        tables_dict[table_idx]["columns"].append(f"{col_name} ({col_type})")

    # Format output
    for table_idx in sorted(tables_dict.keys()):
        table_info = tables_dict[table_idx]
        lines.append(f"  - {table_info['name']}: {', '.join(table_info['columns'])}")

    return "\n".join(lines)


def load_examples(train_spider_path, train_others_path=None, dev_path=None):
    """Load training examples from train_spider.json, train_others.json, and optionally dev.json"""
    examples = []

    print(f"Loading examples from {train_spider_path}...")
    with open(train_spider_path, "r", encoding="utf-8") as f:
        train_spider = json.load(f)
    examples.extend(train_spider)
    print(f"Loaded {len(train_spider)} examples from train_spider.json")

    if train_others_path and os.path.exists(train_others_path):
        print(f"Loading examples from {train_others_path}...")
        with open(train_others_path, "r", encoding="utf-8") as f:
            train_others = json.load(f)
        examples.extend(train_others)
        print(f"Loaded {len(train_others)} examples from train_others.json")

    if dev_path and os.path.exists(dev_path):
        print(f"Loading examples from {dev_path}...")
        with open(dev_path, "r", encoding="utf-8") as f:
            dev_examples = json.load(f)
        examples.extend(dev_examples)
        print(f"Loaded {len(dev_examples)} examples from dev.json")

    print(f"Total examples: {len(examples)}")
    return examples


def process_spider_official(
    spider_dir, output_path, include_schemas=True, include_examples=True
):
    """
    Process official Spider dataset with full schema information.

    Args:
        spider_dir: Path to extracted Spider directory
        output_path: Path to save preprocessed JSON
        include_schemas: Whether to include schema entries
        include_examples: Whether to include example question-SQL pairs
    """
    all_entries = []

    # Load schemas
    tables_path = os.path.join(spider_dir, "tables.json")
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"tables.json not found at {tables_path}")

    schemas = load_schemas(tables_path)

    # Process schemas
    if include_schemas:
        print("\nProcessing database schemas...")
        for schema in schemas:
            schema_text = format_schema_text(schema)

            entry = {
                "text": schema_text,
                "meta": {
                    "type": "schema",
                    "db_id": schema["db_id"],
                    "tables": schema.get(
                        "table_names_original", schema.get("table_names", [])
                    ),
                    "schema": schema_text,
                },
            }
            all_entries.append(entry)

        print(f"Generated {len(schemas)} schema entries")

    # Process examples
    if include_examples:
        print("\nProcessing question-SQL examples...")
        train_spider_path = os.path.join(spider_dir, "train_spider.json")
        train_others_path = os.path.join(spider_dir, "train_others.json")
        dev_path = os.path.join(spider_dir, "dev.json")

        if not os.path.exists(train_spider_path):
            raise FileNotFoundError(
                f"train_spider.json not found at {train_spider_path}"
            )

        examples = load_examples(train_spider_path, train_others_path, dev_path)

        for example in examples:
            question = example["question"]
            sql = example["query"]
            db_id = example["db_id"]

            example_text = f"Question: {question}\nSQL: {sql}"

            entry = {
                "text": example_text,
                "meta": {
                    "type": "example",
                    "db_id": db_id,
                    "question": question,
                    "sql": sql,
                },
            }
            all_entries.append(entry)

        print(f"Generated {len(examples)} example entries")

    print(f"\nTotal entries: {len(all_entries)}")

    # Save to output
    print(f"Saving to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_entries, f, indent=2, ensure_ascii=False)

    print(
        f"✓ Preprocessing complete! Saved {len(all_entries)} entries to {output_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Process official Spider dataset with full schema information"
    )
    parser.add_argument(
        "--spider-dir",
        required=True,
        help="Path to extracted Spider directory (containing tables.json, train_spider.json, etc.)",
    )
    parser.add_argument(
        "--output",
        default="spider_official_processed.json",
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

    args = parser.parse_args()

    include_schemas = not args.examples_only
    include_examples = not args.schemas_only

    process_spider_official(
        args.spider_dir, args.output, include_schemas, include_examples
    )


if __name__ == "__main__":
    main()
