#!/usr/bin/env python3
"""Task eval for the GLM-5.3 quant A/B: 30 math + 15 code + 15 schema tasks,
greedy decoding, deterministic grading.

Usage: task_eval.py HOST PORT --lane LABEL --out FILE [--concurrency N]
Grading: math = last 'Answer: <int>' matches the key; code = extract the
last ```python block, exec, run the task's asserts; schema = parse the
reply as JSON, check key set and the 'must' values (types for the rest).
"""
import argparse, json, re, sys, time
import concurrent.futures
import urllib.request

def chat(host, port, model, prompt, max_tokens):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(f"http://{host}:{port}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.load(r)
    msg = d["choices"][0]["message"]
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    return content, reasoning, d["usage"], time.time() - t0

def grade_math(task, content, reasoning):
    blob = content + "\n" + reasoning
    hits = re.findall(r"Answer:\s*(-?\d+(?:\.\d+)?)", blob)
    if not hits: return False, "no Answer: line"
    got = hits[-1].replace(",", "")
    try: got_n = float(got)
    except ValueError: return False, f"unparseable {got!r}"
    ok = abs(got_n - task["answer"]) < 1e-6
    return ok, "" if ok else f"got {got}, want {task['answer']}"

def grade_code(task, content, reasoning):
    blocks = re.findall(r"```(?:python)?\s*\n(.*?)```", content, re.S)
    if not blocks: return False, "no code block"
    src = blocks[-1]
    ns = {}
    try:
        exec(compile(src, task["id"], "exec"), ns)
    except Exception as e:
        return False, f"exec: {type(e).__name__}: {e}"
    for t in task["tests"]:
        try:
            exec(compile(t, task["id"]+":test", "exec"), ns)
        except Exception as e:
            return False, f"assert failed: {t[:60]} ({type(e).__name__}: {e})"
    return True, ""

def grade_schema(task, content, reasoning):
    blob = content.strip()
    m = re.search(r"\{.*\}", blob, re.S)
    if not m: return False, "no JSON object"
    try:
        obj = json.loads(m.group(0))
    except Exception as e:
        return False, f"json: {e}"
    if not isinstance(obj, dict): return False, "not an object"
    if set(obj) != set(task["types"]): return False, f"keys {sorted(obj)} != {sorted(task['types'])}"
    for k, want in task["must"].items():
        got = obj.get(k)
        if isinstance(want, float):
            if not isinstance(got, (int, float)) or abs(got - want) > 1e-6: return False, f"{k}: {got!r} != {want!r}"
        elif got != want: return False, f"{k}: {got!r} != {want!r}"
    tmap = {"str": str, "int": int, "float": (int, float), "bool": bool, "list": list, "dict": dict}
    for k, tname in task["types"].items():
        if k in task["must"] and not isinstance(task["must"][k], (dict, list)):
            if task["must"][k] is not None and not isinstance(obj[k], (type(task["must"][k]),)): pass
        if tname == "int" and isinstance(obj[k], bool): return False, f"{k}: bool for int"
        if not isinstance(obj[k], tmap[tname]): return False, f"{k}: {type(obj[k]).__name__} not {tname}"
    return True, ""

GRADERS = {"math": grade_math, "code": grade_code, "schema": grade_schema}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("host"); ap.add_argument("port", type=int)
    ap.add_argument("--lane", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="glm-5.3")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    repo = __import__("pathlib").Path(__file__).resolve().parent.parent
    tasks = json.load(open(repo / "benchmarks/results/2026-10-12-glm53-quant-ab/task-eval/prompts.json"))["tasks"]
    def run(task):
        try:
            content, reasoning, usage, wall = chat(a.host, a.port, a.model, task["prompt"], a.max_tokens)
            ok, why = GRADERS[task["cls"]](task, content, reasoning)
            return {"id": task["id"], "cls": task["cls"], "pass": ok, "why": why,
                    "tokens": usage.get("completion_tokens"), "wall_s": round(wall, 2),
                    "content": content[:1500]}
        except Exception as e:
            return {"id": task["id"], "cls": task["cls"], "pass": False, "why": f"error: {e}", "content": ""}
    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        for r in ex.map(run, tasks): rows.append(r)
    summary = {}
    for cls in ("math", "code", "schema"):
        sub = [r for r in rows if r["cls"] == cls]
        summary[cls] = {"n": len(sub), "pass": sum(r["pass"] for r in sub)}
    summary["total"] = sum(s["pass"] for s in summary.values())
    json.dump({"lane": a.lane, "model": a.model, "summary": summary, "rows": rows},
              open(a.out, "w"), indent=1)
    print(f"== {a.lane}: math {summary['math']['pass']}/{summary['math']['n']}"
          f"  code {summary['code']['pass']}/{summary['code']['n']}"
          f"  schema {summary['schema']['pass']}/{summary['schema']['n']}"
          f"  TOTAL {summary['total']}/60")
    for r in rows:
        if not r["pass"]: print(f"  FAIL {r['id']}: {r['why'][:100]}")

if __name__ == "__main__":
    main()
