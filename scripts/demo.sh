#!/usr/bin/env bash
set -euo pipefail

BASE_URL="http://localhost:8000"
REQ_ID() { echo "$(uuidgen 2>/dev/null || python3 -c 'import uuid; print(uuid.uuid4())')"; }

echo "=== RAG Hackathon Demo ==="
echo ""

# 1. Ingest a document
echo ">>> Step 1: Ingest document"
DOC_ID="demo-doc-$(date +%s)"
INGEST=$(curl -s -X POST "${BASE_URL}/ingest" \
  -H "X-Request-Id: $(REQ_ID)" \
  -F "file=@${1:-samples/example.pdf}" \
  -F "doc_id=${DOC_ID}")
JOB_ID=$(echo "$INGEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
VERSION_ID=$(echo "$INGEST" | python3 -c "import sys,json; print(json.load(sys.stdin)['version_id'])")
echo "    doc_id:    ${DOC_ID}"
echo "    job_id:    ${JOB_ID}"
echo "    version:   ${VERSION_ID}"
echo ""

# 2. Poll until done
echo ">>> Step 2: Poll job status"
for i in $(seq 1 30); do
  STATUS=$(curl -s "${BASE_URL}/jobs/${JOB_ID}" -H "X-Request-Id: $(REQ_ID)")
  STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  STAGE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['stage'])")
  PROGRESS=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['progress'])")
  echo "    [${i}/30] state=${STATE} stage=${STAGE} progress=${PROGRESS}%"
  if [ "$STATE" = "done" ]; then
    echo "    Ingest complete!"
    break
  fi
  if [ "$STATE" = "failed" ]; then
    ERROR=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin).get('error','unknown'))")
    echo "    Ingest FAILED: ${ERROR}"
    exit 1
  fi
  sleep 2
done
echo ""

# 3. Query
echo ">>> Step 3: Query the document"
QUERY_RESULT=$(curl -s -X POST "${BASE_URL}/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: $(REQ_ID)" \
  -d "{\"query\": \"What is this document about?\", \"doc_ids\": [\"${DOC_ID}\"]}")
ANSWER=$(echo "$QUERY_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['answer'])")
N_CITATIONS=$(echo "$QUERY_RESULT" | python3 -c "import sys,json; print(len(json.load(sys.stdin)['citations']))")
TIMINGS=$(echo "$QUERY_RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin)['timings_ms']; print(', '.join(f'{k}={v}ms' for k,v in d.items()))")
echo "    Answer: ${ANSWER:0:200}..."
echo "    Citations: ${N_CITATIONS}"
echo "    Timings: ${TIMINGS}"
echo ""

# 4. Run eval
echo ">>> Step 4: Run RAGAS evaluation"
EVAL_RESULT=$(curl -s -X POST "${BASE_URL}/eval/run" \
  -H "X-Request-Id: $(REQ_ID)")
TOTAL=$(echo "$EVAL_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['total_questions'])")
FAILED=$(echo "$EVAL_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['failed_questions'])")
ELAPSED=$(echo "$EVAL_RESULT" | python3 -c "import sys,json; print(json.load(sys.stdin)['elapsed_seconds'])")
AGG=$(echo "$EVAL_RESULT" | python3 -c "
import sys, json
d = json.load(sys.stdin)['aggregate']
for k, v in d.items():
    print(f'    {k}: {v:.4f}')
")
echo "    Questions: ${TOTAL} (failed: ${FAILED})"
echo "    Elapsed: ${ELAPSED}s"
echo "    Aggregate scores:"
echo "${AGG}"
echo ""

# 5. Re-ingest (update)
echo ">>> Step 5: Re-ingest document (new version)"
UPDATE=$(curl -s -X PUT "${BASE_URL}/documents/${DOC_ID}" \
  -H "X-Request-Id: $(REQ_ID)" \
  -F "file=@${1:-samples/example.pdf}")
NEW_JOB=$(echo "$UPDATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
NEW_VER=$(echo "$UPDATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['version_id'])")
echo "    new version: ${NEW_VER}"
echo ""

echo ">>> Step 5b: Poll re-ingest"
for i in $(seq 1 30); do
  STATUS=$(curl -s "${BASE_URL}/jobs/${NEW_JOB}" -H "X-Request-Id: $(REQ_ID)")
  STATE=$(echo "$STATUS" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  if [ "$STATE" = "done" ]; then
    echo "    Re-ingest complete!"
    break
  fi
  if [ "$STATE" = "failed" ]; then
    echo "    Re-ingest FAILED"
    exit 1
  fi
  sleep 2
done
echo ""

# 6. Query again (should get v2)
echo ">>> Step 6: Query again (should use new version)"
QUERY2=$(curl -s -X POST "${BASE_URL}/api/v1/query" \
  -H "Content-Type: application/json" \
  -H "X-Request-Id: $(REQ_ID)" \
  -d "{\"query\": \"What is this document about?\", \"doc_ids\": [\"${DOC_ID}\"]}")
V2_VER=$(echo "$QUERY2" | python3 -c "
import sys, json
cites = json.load(sys.stdin)['citations']
print(cites[0]['version_id'] if cites else 'no citations')
")
echo "    Active version in citations: ${V2_VER}"
echo "    Matches new version: $([ "$V2_VER" = "$NEW_VER" ] && echo 'YES' || echo 'NO')"
echo ""

# 7. Soft delete
echo ">>> Step 7: Soft delete document"
DEL=$(curl -s -X DELETE "${BASE_URL}/documents/${DOC_ID}?mode=soft" \
  -H "X-Request-Id: $(REQ_ID)")
AFFECTED=$(echo "$DEL" | python3 -c "import sys,json; print(json.load(sys.stdin)['points_affected'])")
echo "    Points affected: ${AFFECTED}"
echo ""

echo "=== Demo complete! ==="
