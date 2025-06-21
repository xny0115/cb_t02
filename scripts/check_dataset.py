diff --git a//dev/null b/scripts/check_dataset.py
index 0000000000000000000000000000000000000000..1696bf3cca4a12847913dc6cf0219f1513d1199c 100644
--- a//dev/null
+++ b/scripts/check_dataset.py
@@ -0,0 +1,58 @@
+from __future__ import annotations
+
+import json
+from pathlib import Path
+from typing import List
+
+from src.data.morph import analyze
+
+
+def _load(path: Path) -> List[dict]:
+    with open(path, encoding="utf-8") as f:
+        return json.load(f)
+
+
+def check_file(path: Path) -> int:
+    data = _load(path)
+    errors = 0
+    for idx, item in enumerate(data):
+        q = item.get("question", {})
+        a = item.get("answer", {})
+        if q.get("domain") != item.get("domain"):
+            print(f"domain mismatch at {idx}")
+            errors += 1
+        for field in (q, a):
+            for tok in field.get("tokens", []):
+                if not {"text", "lemma", "pos"} <= tok.keys():
+                    print(f"token missing fields at {idx}")
+                    errors += 1
+                    break
+        # verify analyzer output matches stored tokens
+        if analyze(q.get("text", "")) != q.get("tokens"):
+            print(f"token mismatch at {idx} question")
+            errors += 1
+        if analyze(a.get("text", "")) != a.get("tokens"):
+            print(f"token mismatch at {idx} answer")
+            errors += 1
+        concepts = item.get("concepts", [])
+        if concepts != sorted(set(concepts)):
+            print(f"concepts not sorted/unique at {idx}")
+            errors += 1
+    return errors
+
+
+if __name__ == "__main__":
+    import argparse
+
+    p = argparse.ArgumentParser()
+    p.add_argument("path", type=Path, nargs="?", default=Path("datas"))
+    args = p.parse_args()
+
+    paths = list(args.path.glob("*.json")) if args.path.is_dir() else [args.path]
+    total = 0
+    for fp in paths:
+        total += check_file(fp)
+    if total:
+        print(f"found {total} issues")
+    else:
+        print("all checks passed")
