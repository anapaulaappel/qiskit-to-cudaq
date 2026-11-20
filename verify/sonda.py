import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from runner import executa
casos = {
"laco_e_mz": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(3)\n    h(q[0])\n    for i in range(2):\n        x.ctrl(q[i], q[i+1])\n    mz(q)\nprint(cudaq.sample(kernel))\n",
"angulo_math": "import cudaq, math\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    h(q[0])\n    h(q[1])\n    r1.ctrl(math.pi / 2, q[0], q[1])\n",
"angulo_laco": "import cudaq, numpy as np\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(3)\n    for j in range(3):\n        h(q[j])\n        for k in range(j):\n            r1.ctrl(np.pi / 2 ** (j - k), q[k], q[j])\n",
"ry_ctrl_calc": "import cudaq, numpy as np\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    x(q[0])\n    ry.ctrl(2 * np.arccos(np.sqrt(1 / 2)), q[0], q[1])\n",
"multi_ctrl_front": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(3)\n    x(q)\n    x.ctrl(q.front(2), q[2])\n",
"multi_ctrl_args": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(3)\n    x(q)\n    z.ctrl(q[0], q[1], q[2])\n",
"h_no_vetor": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    h(q)\n",
"lista_angulos": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    t = [0.1, 0.2]\n    for i in range(2):\n        ry(t[i], q[i])\n",
"sem_kernel": "import cudaq\ndef kernel():\n    pass\n",
"sintaxe": "def (:\n",
"porta_inexistente": "import cudaq\n@cudaq.kernel\ndef kernel():\n    q = cudaq.qvector(2)\n    cnot(q[0], q[1])\n",
"exige_args": "import cudaq\n@cudaq.kernel\ndef kernel(n: int):\n    q = cudaq.qvector(n)\n    h(q)\n",
}
for nome, src in casos.items():
    t = time.time(); r, psi = executa(src)
    print(f"{nome:18s} {r['status']:16s} {time.time()-t:4.1f}s", None if psi is None else np.round(psi[:8], 3).tolist(), (r.get('erro') or '')[:90])
