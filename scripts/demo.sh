#!/bin/bash

echo "=== PROVENANCE GUARD DEMO ==="
echo ""

# Moment 1: Basic submission
echo "1. TEXT SUBMISSION WITH STRUCTURED RESPONSE"
echo "==========================================="
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "okay so i finally tried that new ramen place and honestly it was underwhelming. the broth was fine but they put way too much sodium in it.", "creator_id": "demo-1"}' | python -m json.tool
echo ""

# Moment 2: Transparency label
echo "2. TRANSPARENCY LABEL (What Readers See)"
echo "========================================"
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The afternoon light fell across my desk. I watched the dust motes drift through golden bars of light. The coffee had gone cold.", "creator_id": "demo-2"}' | python -c "import sys, json; d = json.load(sys.stdin); print(d['label']['text'])"
echo ""

# Moment 3: Confidence differences
echo "3. CONFIDENCE SCORE DIFFERENCES"
echo "==============================="
echo "Clearly human-written text (informal, messy, personal):"
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "ok so last night was INSANE. we went to this bar downtown right and there was like no one there. the bartender kept giving us weird looks. my friend sarah ordered this weird drink that tasted like battery acid lol. we left after like 20 minutes. worst night ever honestly", "creator_id": "demo-clear-human"}' | python -c "import sys, json; d = json.load(sys.stdin); print(f\"Score: {d['combined_score']:.2f}, Confidence: {d['confidence']:.2f}, Verdict: {d['verdict']}\")"

echo ""
echo "Clearly AI-generated text (uniform, hedged, formal):"
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The implementation of machine learning algorithms has demonstrated significant potential in addressing contemporary challenges. It is important to acknowledge that while the applications are diverse, careful consideration of methodological approaches is necessary. Stakeholders must collaborate effectively to ensure that deployment strategies align with established frameworks and best practices. Furthermore, continued evaluation of outcomes is essential to validate effectiveness and optimize results.", "creator_id": "demo-clear-ai"}' | python -c "import sys, json; d = json.load(sys.stdin); print(f\"Score: {d['combined_score']:.2f}, Confidence: {d['confidence']:.2f}, Verdict: {d['verdict']}\")"

echo ""
echo "Ambiguous text (borderline case):"
curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "Remote work has fundamentally changed how many people approach their professional responsibilities. There are clear advantages to flexible scheduling and reduced commuting time. However, the lack of face-to-face interaction can create challenges in team communication and company culture. Organizations must carefully balance these competing interests.", "creator_id": "demo-ambiguous"}' | python -c "import sys, json; d = json.load(sys.stdin); print(f\"Score: {d['combined_score']:.2f}, Confidence: {d['confidence']:.2f}, Verdict: {d['verdict']}\")"
echo ""

# Moment 4: Seed dashboard data directly into SQLite (bypasses rate limiter)
# Target: ~15% likely_ai, ~38% likely_human, ~47% uncertain across 99 rows.
# Run with --clear to wipe previous seed rows before re-seeding.
echo "4. SEEDING DASHBOARD DATA (99 rows: 16 AI + 40 human + 43 uncertain)"
echo "======================================================================"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
python "$SCRIPT_DIR/seed_dashboard.py" --clear
echo ""

# Moment 5: Appeal workflow
echo "5. APPEAL WORKFLOW"
echo "=================="
CONTENT_ID=$(curl -s -X POST http://localhost:5000/submit \
  -H "Content-Type: application/json" \
  -d '{"text": "The morning broke grey and cold. I sat with coffee, watching rain stream down the window.", "creator_id": "demo-5"}' | python -c "import sys, json; print(json.load(sys.stdin)['content_id'])")

echo "Submitting appeal for content ID: $CONTENT_ID"
curl -s -X POST http://localhost:5000/appeal \
  -H "Content-Type: application/json" \
  -d "{\"content_id\": \"$CONTENT_ID\", \"reasoning\": \"I wrote this myself. I am a non-native English speaker writing formally.\"}" | python -m json.tool

echo "Content record with appeal:"
curl -s http://localhost:5000/content/$CONTENT_ID | python -m json.tool
echo ""

# Moment 6: Rate limiting
echo "6. RATE LIMITING BEHAVIOR"
echo "========================="
echo "Sending 12 requests (limit is 10/hour)..."
for i in $(seq 1 12); do
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST http://localhost:5000/submit \
    -H "Content-Type: application/json" \
    -d '{"text": "Rate limit test submission.", "creator_id": "ratelimit"}')
  echo "Request $i: $STATUS"
done
echo ""

# Moment 7: Audit log
echo "7. AUDIT LOG WITH 3+ ENTRIES"
echo "============================"
curl -s http://localhost:5000/log | python -m json.tool
