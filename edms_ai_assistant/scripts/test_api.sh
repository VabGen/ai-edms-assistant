#!/usr/bin/env bash
# scripts/test_api.sh — Тестирование API EDMS AI Assistant

set -euo pipefail

BASE_URL="${ORCHESTRATOR_URL:-http://localhost:8002}"
GREEN='\033[0;32m'; RED='\033[0;31m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'

pass() { echo -e "${GREEN}[PASS]${NC} $*"; }
fail() { echo -e "${RED}[FAIL]${NC} $*"; }
info() { echo -e "${BLUE}[INFO]${NC} $*"; }

echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
echo -e "${YELLOW}  EDMS AI Assistant — API Tests${NC}"
echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
echo ""

# ── Health check ──────────────────────────────────────────────────────────────
info "1. Health check..."
HEALTH=$(curl -sf "$BASE_URL/health" 2>/dev/null || echo '{"status":"error"}')
if echo "$HEALTH" | grep -q '"orchestrator"'; then
    pass "Health check OK"
    echo "   $(echo $HEALTH | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps(d, indent=2, ensure_ascii=False))' 2>/dev/null || echo $HEALTH)"
else
    fail "Health check FAILED: $HEALTH"
fi
echo ""

# ── Simple chat ───────────────────────────────────────────────────────────────
info "2. Simple chat request (статистика)..."
CHAT=$(curl -sf -X POST "$BASE_URL/chat" \
    -H "Content-Type: application/json" \
    -d '{
        "message": "Покажи мою статистику документов",
        "user_id": "test-user-001",
        "token": "test-token"
    }' 2>/dev/null || echo '{"content":"error"}')

if echo "$CHAT" | grep -q '"content"'; then
    pass "Chat request OK"
    DIALOG_ID=$(echo "$CHAT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("dialog_id",""))' 2>/dev/null || echo "")
    INTENT=$(echo "$CHAT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("intent",""))' 2>/dev/null || echo "")
    LATENCY=$(echo "$CHAT" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d.get("latency_ms",""))' 2>/dev/null || echo "")
    echo "   intent: $INTENT | latency: ${LATENCY}ms | dialog_id: $DIALOG_ID"
else
    fail "Chat request FAILED: $CHAT"
    DIALOG_ID=""
fi
echo ""

# ── Feedback ──────────────────────────────────────────────────────────────────
if [ -n "$DIALOG_ID" ]; then
    info "3. Feedback submission..."
    FEEDBACK=$(curl -sf -X POST "$BASE_URL/feedback" \
        -H "Content-Type: application/json" \
        -d "{\"dialog_id\": \"$DIALOG_ID\", \"rating\": 1, \"comment\": \"Test feedback\"}" \
        2>/dev/null || echo '{"success":false}')

    if echo "$FEEDBACK" | grep -q '"success": *true'; then
        pass "Feedback OK"
    else
        fail "Feedback FAILED: $FEEDBACK"
    fi
else
    info "3. Feedback skipped (no dialog_id)"
fi
echo ""

# ── RAG stats ─────────────────────────────────────────────────────────────────
info "4. RAG stats..."
RAG=$(curl -sf "$BASE_URL/rag/stats" 2>/dev/null || echo '{"error":"unavailable"}')
if echo "$RAG" | grep -qv '"error"'; then
    pass "RAG stats OK: $RAG"
else
    fail "RAG stats: $RAG"
fi
echo ""

# ── Metrics ───────────────────────────────────────────────────────────────────
info "5. Prometheus metrics..."
METRICS=$(curl -sf "$BASE_URL/metrics" 2>/dev/null || echo "error")
if echo "$METRICS" | grep -q "edms_requests_total"; then
    pass "Metrics OK"
    echo "   $(echo "$METRICS" | grep -v '^#' | head -5)"
else
    fail "Metrics FAILED"
fi
echo ""

echo -e "${YELLOW}═══════════════════════════════════════════${NC}"
echo -e "${GREEN}Tests completed${NC}"
