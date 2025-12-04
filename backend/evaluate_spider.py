import json, os, argparse, sqlite3, requests, time


def run_sql(db_file: str, sql: str):
    con = sqlite3.connect(db_file)
    cur = con.cursor()
    cur.execute(sql)
    rows = cur.fetchall()
    con.close()
    return rows


def norm(rows):
    # order-insensitive compare, keeps duplicates via sorting
    return sorted(rows)


def main(split_path, db_root, backend_url, out_path, top_k=6, limit=None, sleep=0.0):
    data = json.load(open(split_path, "r", encoding="utf-8"))
    if limit:
        data = data[:limit]

    total = len(data)
    correct = 0
    pred_sql_errors = 0
    pred_exec_errors = 0
    gold_exec_errors = 0
    api_errors = 0

    results = []

    for i, ex in enumerate(data, 1):
        q = ex["question"]
        db_id = ex["db_id"]
        gold_sql = ex.get("query", "")

        db_file = os.path.join(db_root, db_id, f"{db_id}.sqlite")
        if not os.path.exists(db_file):
            # some spider dumps use .db instead of .sqlite
            alt = os.path.join(db_root, db_id, f"{db_id}.db")
            db_file = alt if os.path.exists(alt) else db_file

        pred_sql = ""
        api_error_text = ""

        # 1) call backend
        try:
            r = requests.post(
                f"{backend_url}/generate",
                json={"question": q, "db_id": db_id, "top_k": top_k},
                timeout=60,
            )
            if r.status_code == 200:
                result = r.json()
                pred_sql = result.get("generated_sql", "")
                # Strip whitespace and ensure it's a string
                if pred_sql:
                    pred_sql = pred_sql.strip()
            else:
                api_errors += 1
                api_error_text = r.text
        except Exception as e:
            api_errors += 1
            api_error_text = str(e)

        # 2) execute gold + pred and compare
        is_correct = False
        gold_rows = None
        pred_rows = None
        pred_err = ""
        gold_err = ""

        # run gold
        try:
            gold_rows = norm(run_sql(db_file, gold_sql))
        except Exception as e:
            gold_exec_errors += 1
            gold_err = str(e)

        # run pred (only if we got a pred and gold ran)
        if pred_sql and pred_sql.strip() and gold_rows is not None:
            try:
                pred_rows = norm(run_sql(db_file, pred_sql))
                is_correct = pred_rows == gold_rows
            except sqlite3.OperationalError as e:
                pred_sql_errors += 1
                pred_err = str(e)
            except Exception as e:
                pred_exec_errors += 1
                pred_err = str(e)
        elif not pred_sql and not api_error_text:
            # Empty SQL returned but no API error
            pred_sql_errors += 1
            pred_err = "Empty SQL returned from backend"

        if is_correct:
            correct += 1

        results.append(
            {
                "i": i,
                "db_id": db_id,
                "question": q,
                "gold_sql": gold_sql,
                "pred_sql": pred_sql,
                "correct_exec": is_correct,
                "api_error": api_error_text,
                "gold_error": gold_err,
                "pred_error": pred_err,
            }
        )

        if i % 50 == 0:
            acc_so_far = correct / i if i > 0 else 0.0
            print(
                f"[{i}/{total}] exec_acc={acc_so_far:.3f} | correct={correct} | api_err={api_errors} | pred_sql_err={pred_sql_errors}"
            )

        if sleep:
            time.sleep(sleep)

    summary = {
        "total": total,
        "exec_correct": correct,
        "exec_acc": (correct / total) if total else 0.0,
        "api_error_rate": api_errors / total if total else 0.0,
        "pred_sql_error_rate": pred_sql_errors / total if total else 0.0,
        "pred_exec_error_rate": pred_exec_errors / total if total else 0.0,
        "gold_exec_error_rate": gold_exec_errors / total if total else 0.0,
        "backend_url": backend_url,
        "db_root": db_root,
        "split_path": split_path,
        "top_k": top_k,
    }

    out = {"summary": summary, "results": results}
    json.dump(out, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print("saved:", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="spider_data/test.json or dev.json")
    ap.add_argument(
        "--db_root",
        required=True,
        help="spider_data/database OR spider_data/test_database",
    )
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=0.0)
    args = ap.parse_args()
    main(
        args.split,
        args.db_root,
        args.backend,
        args.out,
        args.top_k,
        args.limit,
        args.sleep,
    )
