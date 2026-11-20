#!/usr/bin/env bash
# Envia um arquivo do Mac para o pod do vLLM (quantum-finetune, app=vllm-nemotron) com verificação:
# pedaços de 4 MB, sha256 por pedaço (repete se falhar) e checksum final. O fluxo do `oc exec` neste
# cluster corta transferências grandes, por isso os pedaços.
#   uso: k8s/empurra_para_vllm.sh <arquivo no Mac> <caminho no pod do vLLM>
set -euo pipefail
LOCAL="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; REMOTO="$2"; NS=quantum-finetune
POD=$(oc get pod -n "$NS" -l app=vllm-nemotron -o jsonpath='{.items[0].metadata.name}')
ex() { oc exec -n "$NS" "$POD" -- "$@"; }
TMP=$(mktemp -d); (cd "$TMP" && split -b 4M "$LOCAL" p_)
ex bash -c "mkdir -p '$(dirname "$REMOTO")' && : > '$REMOTO.part'"
for p in "$TMP"/p_*; do
  soma=$(shasum -a 256 "$p" | cut -d' ' -f1)
  for tentativa in 1 2 3 4 5 6; do
    oc exec -i -n "$NS" "$POD" -- bash -c "cat > /tmp/_pedaco" < "$p"
    [ "$(ex sha256sum /tmp/_pedaco | cut -d' ' -f1)" = "$soma" ] && break
    [ "$tentativa" = 6 ] && { echo "FALHOU: $p"; exit 1; }
  done
  ex bash -c "cat /tmp/_pedaco >> '$REMOTO.part'"
done
TOTAL=$(shasum -a 256 "$LOCAL" | cut -d' ' -f1)
[ "$(ex sha256sum "$REMOTO.part" | cut -d' ' -f1)" = "$TOTAL" ] || { echo "FALHOU: checksum final"; exit 1; }
ex bash -c "mv '$REMOTO.part' '$REMOTO' && rm -f /tmp/_pedaco"; rm -rf "$TMP"
echo "ok $REMOTO ($TOTAL)"
