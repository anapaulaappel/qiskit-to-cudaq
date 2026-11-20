"""Prepara o código candidato para get_state. Tudo aqui é declarado no README.

Mantém só definições: imports, funções, classes e constantes. Descarta o que estiver fora
disso (print, cudaq.sample, set_target), porque o alvo e a execução são do verificador.
Remove chamadas a mz: get_state em kernel com mz devolve um estado colapsado e aleatório,
então um kernel correto reprovaria por sorte (ver a calibração no README).
"""
import ast


def _chama(no, nome):
    return isinstance(no, ast.Call) and isinstance(no.func, ast.Name) and no.func.id == nome


def _usa_cudaq(no):
    return any(isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
               and n.value.id == "cudaq" and n.attr in ("sample", "observe", "get_state", "run")
               for n in ast.walk(no))


class _TiraMz(ast.NodeTransformer):
    def _bloco(self, no):
        self.generic_visit(no)
        return no

    def visit_Expr(self, no):
        return None if _chama(no.value, "mz") else no

    def visit_Assign(self, no):
        return None if _chama(no.value, "mz") else no

    def generic_visit(self, no):
        super().generic_visit(no)
        if isinstance(no, (ast.FunctionDef, ast.For, ast.While, ast.If, ast.With)) and not no.body:
            no.body = [ast.Pass()]
        return no


def sanitiza(fonte):
    """Devolve o fonte limpo. Levanta SyntaxError se o código não for Python válido."""
    arvore = ast.parse(fonte)
    mantidos = []
    for no in arvore.body:
        if isinstance(no, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef)):
            mantidos.append(no)
        elif isinstance(no, (ast.Assign, ast.AnnAssign)) and not _usa_cudaq(no):
            mantidos.append(no)
    arvore.body = [_TiraMz().visit(no) for no in mantidos]
    return ast.unparse(arvore) + "\n"
