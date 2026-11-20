#!/usr/bin/env bash
# Copia um arquivo grande do workbench (quantum-finetune/training-0) para o Mac com verificação.
# O fluxo do `oc exec`/`oc cp` neste cluster corta transferências grandes (~20–70 MB), então o arquivo
# é dividido em pedaços de 4 MB, cada pedaço é conferido por sha256 e repetido se falhar, e o arquivo
# final é conferido de novo.
#   uso: k8s/puxa_do_workbench.sh <caminho no workbench> <destino no Mac>
set -euo pipefail
REMOTO="$1"; LOCAL="$2"
NS=quantum-finetune; POD=training-0; C=training
ex() { oc exec -n "$NS" "$POD" -c "$C" -- "$@"; }
TMP=$(ex mktemp -d); ex bash -c "cd $TMP && split -b 4M '$REMOTO' p_ && sha256sum p_* > sums"
mkdir -p "$(dirname "$LOCAL")"; : > "$LOCAL.part"
while read -r soma nome; do
  for tentativa in 1 2 3 4 5 6; do
    ex cat "$TMP/$nome" > /tmp/_pedaco
    [ "$(shasum -a 256 /tmp/_pedaco | cut -d' ' -f1)" = "$soma" ] && break
    [ "$tentativa" = 6 ] && { echo "FALHOU: $nome"; exit 1; }
  done
  cat /tmp/_pedaco >> "$LOCAL.part"
done < <(ex cat "$TMP/sums")
TOTAL=$(ex sha256sum "$REMOTO" | cut -d' ' -f1)
[ "$(shasum -a 256 "$LOCAL.part" | cut -d' ' -f1)" = "$TOTAL" ] || { echo "FALHOU: checksum final"; exit 1; }
mv "$LOCAL.part" "$LOCAL"; ex rm -rf "$TMP"; rm -f /tmp/_pedaco
echo "ok $LOCAL ($TOTAL)"
