#!/usr/bin/env python3
"""
Octave/MATLAB to Python converter v3
Usage: python oct2py_converter.py input.m -o output.py
"""

import re
import argparse

MATH_FUNCS = {
    "zeros":"np.zeros","ones":"np.ones","eye":"np.eye",
    "rand":"np.random.rand","randn":"np.random.randn",
    "linspace":"np.linspace","logspace":"np.logspace",
    "diag":"np.diag","inv":"np.linalg.inv","det":"np.linalg.det",
    "norm":"np.linalg.norm","eig":"np.linalg.eig","svd":"np.linalg.svd",
    "cross":"np.cross","dot":"np.dot","kron":"np.kron","trace":"np.trace",
    "rank":"np.linalg.matrix_rank","pinv":"np.linalg.pinv",
    "reshape":"np.reshape","repmat":"np.tile",
    "horzcat":"np.hstack","vertcat":"np.vstack",
    "fliplr":"np.fliplr","flipud":"np.flipud","rot90":"np.rot90",
    "sort":"np.sort","unique":"np.unique","find":"np.where",
    "sum":"np.sum","prod":"np.prod","cumsum":"np.cumsum",
    "min":"np.min","max":"np.max","abs":"np.abs",
    "real":"np.real","imag":"np.imag","conj":"np.conj","angle":"np.angle",
    "floor":"np.floor","ceil":"np.ceil","round":"np.round",
    "mod":"np.mod","rem":"np.remainder","sign":"np.sign",
    "sin":"np.sin","cos":"np.cos","tan":"np.tan",
    "asin":"np.arcsin","acos":"np.arccos","atan":"np.arctan","atan2":"np.arctan2",
    "sinh":"np.sinh","cosh":"np.cosh","tanh":"np.tanh",
    "exp":"np.exp","log":"np.log","log2":"np.log2","log10":"np.log10","sqrt":"np.sqrt",
    "size":"np.shape","numel":"np.size","length":"len",
    "any":"np.any","all":"np.all",
    "disp":"print","num2str":"str","str2num":"float","str2double":"float",
    "fft":"np.fft.fft","ifft":"np.fft.ifft","fftshift":"np.fft.fftshift",
    "griddata":"scipy.interpolate.griddata",
    "interp1":"scipy.interpolate.interp1d",
    "meshgrid":"np.meshgrid",
    "linsolve":"np.linalg.solve",
    "mean":"np.mean","std":"np.std","var":"np.var","median":"np.median",
    "cov":"np.cov","corrcoef":"np.corrcoef",
    "conv":"np.convolve",
    "interp2":"scipy.interpolate.interp2d",
    "sparse":"scipy.sparse.csr_matrix",
}

PLOT_FUNCS = {
    "figure":"plt.figure","plot":"plt.plot","plot3":"plt.plot",
    "semilogx":"plt.semilogx","semilogy":"plt.semilogy","loglog":"plt.loglog",
    "scatter":"plt.scatter","bar":"plt.bar","hist":"plt.hist",
    "imagesc":"plt.imshow","pcolor":"plt.pcolormesh","pcolormesh":"plt.pcolormesh",
    "contour":"plt.contour","contourf":"plt.contourf",
    "xlabel":"plt.xlabel","ylabel":"plt.ylabel","title":"plt.title",
    "legend":"plt.legend","colorbar":"plt.colorbar",
    "xlim":"plt.xlim","ylim":"plt.ylim","subplot":"plt.subplot",
    "clf":"plt.clf","close":"plt.close","tight_layout":"plt.tight_layout",
    "clim":"plt.clim",
}

ALL_KNOWN_FUNCS = set(MATH_FUNCS.keys()) | set(PLOT_FUNCS.keys()) | {
    "print","range","int","float","str","len","list","dict",
    "isinstance","enumerate","zip","map","filter","type",
}

SCALAR_SUBS = [
    (r"\bpi\b",    "np.pi"),
    (r"\bInf\b",   "np.inf"),
    (r"\binf\b",   "np.inf"),
    (r"\bNaN\b",   "np.nan"),
    (r"\bnan\b",   "np.nan"),
    (r"\btrue\b",  "True"),
    (r"\bfalse\b", "False"),
    (r"\beps\b",   "np.finfo(float).eps"),
]


class OctaveConverter:
    def __init__(self):
        self.needs_numpy         = False
        self.needs_matplotlib    = False
        self.needs_scipy_io      = False
        self.needs_scipy_interp  = False
        self.needs_fprintf       = False
        self.indent_level        = 0
        # Track loop variables that start from 1 (MATLAB) → 0-based in Python
        # key = varname, value = start offset (1 means MATLAB started at 1)
        self.loop_vars           = {}   # {varname: matlab_start}

    def convert_file(self, path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            src = f.read()
        return self._convert(src)

    def _convert(self, src):
        lines = src.splitlines()
        raw_body = []
        self.indent_level = 0
        for raw in lines:
            out = self._process_line(raw)
            if out is not None:
                raw_body.append(out)

        # Inject 'pass' where a block header (ends with ':') is followed by
        # a line at same or lower indent (empty block)
        body = []
        for i, line in enumerate(raw_body):
            body.append(line)
            stripped = line.rstrip()
            if stripped.endswith(":"):
                # Look ahead for next non-empty line
                next_content = None
                for j in range(i + 1, len(raw_body)):
                    if raw_body[j].strip():
                        next_content = raw_body[j]
                        break
                cur_indent  = len(stripped) - len(stripped.lstrip())
                if next_content is None:
                    body.append(" " * (cur_indent + 4) + "pass")
                else:
                    next_indent = len(next_content) - len(next_content.lstrip())
                    if next_indent <= cur_indent:
                        body.append(" " * (cur_indent + 4) + "pass")

        return "\n".join(self._make_header() + ["", ""] + body)

    def _make_header(self):
        h = ["# Generated by oct2py_converter"]
        if self.needs_numpy:         h.append("import numpy as np")
        if self.needs_matplotlib:    h.append("import matplotlib.pyplot as plt")
        if self.needs_scipy_io:      h.append("import scipy.io")
        if self.needs_scipy_interp:  h.append("import scipy.interpolate")
        if self.needs_fprintf:
            h += ["",
                  "def _fprintf(fmt, *args): print(fmt % args, end='')",
                  "def _sprintf(fmt, *args): return fmt % args"]
        return h

    def _ind(self):
        return "    " * self.indent_level

    def _process_line(self, raw):
        line = raw.rstrip()
        if not line.strip():
            return ""

        # Split on semicolons that separate multiple statements on one line,
        # but only at top level (not inside strings or brackets)
        stmts = self._split_statements(line.strip())

        if len(stmts) == 1:
            code, cmt = self._split_comment(stmts[0])
            cmt_str   = ("  # " + cmt) if cmt else ""
            code      = code.rstrip(";").strip()
            if not code:
                return self._ind() + cmt_str.strip()
            result = self._stmt(code)
            return result + cmt_str if result is not None else None
        else:
            parts = []
            for s in stmts:
                code, cmt = self._split_comment(s.strip())
                code = code.rstrip(";").strip()
                if not code:
                    continue
                result = self._stmt(code)
                if result is not None and result.strip():
                    parts.append(result.strip())
            return self._ind() + "; ".join(parts) if parts else ""

    def _split_statements(self, line):
        """Split line on ; separating statements, not inside strings/brackets."""
        parts, depth_p, depth_b, in_str, sc, cur = [], 0, 0, False, None, ""
        for ch in line:
            if not in_str and ch in ('"', "'"):
                in_str, sc = True, ch
            elif in_str and ch == sc:
                in_str = False
            elif not in_str:
                if ch == "(": depth_p += 1
                elif ch == ")": depth_p -= 1
                elif ch == "[": depth_b += 1
                elif ch == "]": depth_b -= 1
                elif ch == ";" and depth_p == 0 and depth_b == 0:
                    parts.append(cur); cur = ""
                    continue
            cur += ch
        if cur.strip():
            parts.append(cur)
        return [p for p in parts if p.strip()]

    def _split_comment(self, line):
        in_str, sc = False, None
        for i, ch in enumerate(line):
            if not in_str and ch in ('"', "'"):
                in_str, sc = True, ch
            elif in_str and ch == sc:
                in_str = False
            elif not in_str and ch == "%":
                return line[:i].rstrip(), line[i+1:].strip()
        return line, ""

    def _stmt(self, code):
        ind = self._ind()

        if re.match(r"^(clear|clc)\b", code):
            return ind + "# " + code

        if re.fullmatch(r"end", code):
            self.indent_level = max(0, self.indent_level - 1)
            return ""  # caller handles pass injection

        m = re.match(r"^function\s+(?:(\[?[\w\s,]+\]?)\s*=\s*)?(\w+)\s*\(([^)]*)\)", code)
        if m:
            ret, name, args = m.group(1), m.group(2), m.group(3)
            s = ind + f"def {name}({args.strip()}):"
            if ret: s += f"  # returns {ret.strip()}"
            self.indent_level += 1
            return s

        if code == "return":
            return ind + "return"

        m = re.match(r"^for\s+(\w+)\s*=\s*(.+)", code)
        if m:
            var      = m.group(1)
            rng_raw  = m.group(2).strip()
            rng_py, matlab_start = self._range_for(rng_raw)
            # Register this as a loop variable with its MATLAB start value
            self.loop_vars[var] = matlab_start
            s = ind + f"for {var} in {rng_py}:"
            self.indent_level += 1
            return s

        m = re.match(r"^while\s+(.+)", code)
        if m:
            s = ind + f"while {self._expr(m.group(1).strip())}:"
            self.indent_level += 1
            return s

        m = re.match(r"^if\s+(.+)", code)
        if m:
            s = ind + f"if {self._expr(m.group(1).strip())}:"
            self.indent_level += 1
            return s

        m = re.match(r"^elseif\s+(.+)", code)
        if m:
            self.indent_level = max(0, self.indent_level - 1)
            s = self._ind() + f"elif {self._expr(m.group(1).strip())}:"
            self.indent_level += 1
            return s

        if code == "else":
            self.indent_level = max(0, self.indent_level - 1)
            s = self._ind() + "else:"
            self.indent_level += 1
            return s

        if code == "break":    return ind + "break"
        if code == "continue": return ind + "continue"

        # ── save('file.mat', 'a', 'b', ...) ─────────────────────────────────
        m = re.match(r"^save\s*\(\s*['\"](.+?)['\"](.*)?\)", code)
        if m:
            fname    = m.group(1)
            rest     = m.group(2) or ""
            varnames = [v.strip().strip("'\"") for v in rest.split(",") if v.strip().strip("'\"")]
            if fname.endswith(".mat"):
                self.needs_scipy_io = True
                if varnames:
                    vars_dict = "{" + ", ".join(f'"{v}": {v}' for v in varnames) + "}"
                    return ind + f"scipy.io.savemat('{fname}', {vars_dict})"
                else:
                    return ind + f"# save('{fname}')  # specify variables: scipy.io.savemat('{fname}', dict(a=a, b=b))"
            elif fname.endswith(".txt") or fname.endswith(".csv"):
                self.needs_numpy = True
                if len(varnames) == 1:
                    return ind + f"np.savetxt('{fname}', {varnames[0]}, delimiter=' ')"
                elif varnames:
                    combined = f"np.column_stack([{', '.join(varnames)}])"
                    return ind + f"np.savetxt('{fname}', {combined}, delimiter=' ')"
                else:
                    return ind + f"# np.savetxt('{fname}', data)"
            else:
                # generic — treat as .mat
                self.needs_scipy_io = True
                if varnames:
                    vars_dict = "{" + ", ".join(f'"{v}": {v}' for v in varnames) + "}"
                    return ind + f"scipy.io.savemat('{fname}', {vars_dict})"
                return ind + f"# save('{fname}')"

        # ── load('file.mat') / load('file.txt') ──────────────────────────────
        m = re.match(r"^load\s*\(?\s*['\"](.+?)['\"]", code)
        if m:
            fname = m.group(1)
            if fname.endswith(".mat"):
                self.needs_scipy_io = True
                var = re.sub(r"[^a-zA-Z0-9_]", "_", fname.replace(".mat", ""))
                return ind + f"{var} = scipy.io.loadmat('{fname}')"
            elif fname.endswith(".txt") or fname.endswith(".csv"):
                self.needs_numpy = True
                var = re.sub(r"[^a-zA-Z0-9_]", "_", fname.replace(".txt","").replace(".csv",""))
                delim = "','" if fname.endswith(".csv") else "None"
                return ind + f"{var} = np.loadtxt('{fname}', delimiter={delim})"
            else:
                # unknown extension — try loadmat
                self.needs_scipy_io = True
                var = re.sub(r"[^a-zA-Z0-9_]", "_", fname)
                return ind + f"{var} = scipy.io.loadmat('{fname}')"

        m = re.match(r"^figure\s*(\(.*\))?$", code)
        if m:
            self.needs_matplotlib = True
            args = m.group(1) or "()"
            return ind + f"plt.figure{args}"

        if code in ("drawnow", "show"):
            self.needs_matplotlib = True
            return ind + "plt.show()"

        m = re.match(r"^grid\s+(on|off)$", code)
        if m:
            self.needs_matplotlib = True
            val = "True" if m.group(1) == "on" else "False"
            return ind + f"plt.grid({val})"

        m = re.match(r"^axis\s+(equal|off|tight|on)$", code)
        if m:
            self.needs_matplotlib = True
            return ind + f"plt.axis('{m.group(1)}')"

        m = re.match(r"^hold\s+(on|off)$", code)
        if m:
            self.needs_matplotlib = True
            # matplotlib holds by default — no-op, just comment if 'off'
            if m.group(1) == "off":
                return ind + "plt.gca().set_prop_cycle(None)  # hold off"
            return ""  # hold on is default in matplotlib, skip

        # scatter(x, y, sz, c, 'filled') → plt.scatter(x, y, s=sz, c=c, cmap='viridis')
        m = re.match(r"^scatter\s*\((.+)\)$", code)
        if m:
            self.needs_matplotlib = True
            raw_args = self._split_args(m.group(1))
            args_py  = [self._expr(a.strip()) for a in raw_args]
            # Strip 'filled' string arg
            args_py = [a for a in args_py if a.strip() not in ('"filled"', "'filled'", '"filled"')]
            if len(args_py) >= 4:
                x, y, sz, c = args_py[0], args_py[1], args_py[2], args_py[3]
                return ind + f"plt.scatter({x}, {y}, s={sz}, c={c}, cmap='viridis')"
            elif len(args_py) == 3:
                x, y, sz = args_py[0], args_py[1], args_py[2]
                return ind + f"plt.scatter({x}, {y}, s={sz})"
            else:
                x, y = args_py[0], args_py[1]
                return ind + f"plt.scatter({x}, {y})"

        m = re.match(r"^shading\s*\(?\s*['\"]?(\w+)['\"]?\)?$", code)
        if m:
            return ind + f"# shading '{m.group(1)}'"

        m = re.match(r"^fprintf\s*\((.+)\)$", code)
        if m:
            self.needs_fprintf = True
            return ind + f"_fprintf({m.group(1)})"

        # ── eig / eigs ────────────────────────────────────────────────────────
        # [V,D] = eig(A)        → eigenvalues, eigenvectors = np.linalg.eig(A)
        # [V,D] = eig(A,B)      → scipy.linalg.eig generalized
        # [V,D] = eigs(A,k)     → scipy.sparse.linalg.eigs
        m = re.match(r"^\[([^\]]+)\]\s*=\s*(eigs?)\s*\((.+)\)$", code)
        if m:
            lhs_raw  = m.group(1)
            fn       = m.group(2)
            args_raw = self._split_args(m.group(3))
            args_py  = [self._expr(a.strip()) for a in args_raw]
            lhs_vars = [v.strip() for v in lhs_raw.split(",")]
            # Conventionally [V,D] → eigenvalues=D diagonal, eigenvectors=V
            # numpy returns (eigenvalues, eigenvectors)
            ev_var  = lhs_vars[1] if len(lhs_vars) > 1 else "eigenvalues"
            vec_var = lhs_vars[0]
            if fn == "eigs":
                self.needs_scipy_io = True  # triggers scipy import
                self.needs_numpy    = True
                # scipy.sparse.linalg.eigs returns (values, vectors)
                args_str = ", ".join(args_py)
                return (ind + f"import scipy.sparse.linalg\n" +
                        ind + f"{ev_var}, {vec_var} = scipy.sparse.linalg.eigs({args_str})\n" +
                        ind + f"{ev_var} = np.diag(np.real({ev_var}))  # diagonal eigenvalue matrix")
            else:
                self.needs_numpy = True
                if len(args_py) == 2:
                    # Generalized eig(A, B)
                    self.needs_scipy_io = True
                    return (ind + f"import scipy.linalg\n" +
                            ind + f"{ev_var}_vals, {vec_var} = scipy.linalg.eig({args_py[0]}, {args_py[1]})\n" +
                            ind + f"{ev_var} = np.diag(np.real({ev_var}_vals))  # diagonal eigenvalue matrix")
                else:
                    return (ind + f"{ev_var}_vals, {vec_var} = np.linalg.eig({args_py[0]})\n" +
                            ind + f"{ev_var} = np.diag(np.real({ev_var}_vals))  # diagonal eigenvalue matrix")

        m = re.match(r"^\[([^\]]+)\]\s*=\s*(.+)", code)
        if m:
            lhs = ", ".join(v.strip() for v in m.group(1).split(","))
            rhs = self._expr(m.group(2).strip())
            return ind + f"{lhs} = {rhs}"

        m = re.match(r"^([\w.]+(?:\([^)=]+\))?)\s*=\s*(.+)", code)
        if m:
            lhs = self._lhs(m.group(1))
            rhs = self._expr(m.group(2).strip())
            return ind + f"{lhs} = {rhs}"

        for mf, pf in PLOT_FUNCS.items():
            if code.strip() == mf:
                self.needs_matplotlib = True
                return ind + pf + "()"

        return ind + self._expr(code)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _lhs(self, s):
        m = re.match(r"^(\w+)\((.+)\)$", s)
        if m and m.group(1) not in ALL_KNOWN_FUNCS:
            return f"{m.group(1)}[{self._index(m.group(2))}]"
        return s

    def _range_for(self, rng):
        """Convert MATLAB for-range to Python range/arange.
        Returns (python_range_str, matlab_start_value_or_None).
        matlab_start is used to know if loop var needs offset when used as index.
        """
        parts = rng.split(":")
        if len(parts) == 2:
            a_raw = parts[0].strip()
            b_raw = parts[1].strip()
            a = self._expr(a_raw)
            b = self._expr(b_raw)
            # Integer constants
            try:
                ai, bi = int(a_raw), int(b_raw)
                n = bi - ai + 1
                if ai == 0:
                    return f"range({bi + 1})", 0    # 0-based already
                elif ai == 1:
                    return f"range({n})", 1          # range(10) → 0..9
                else:
                    return f"range({ai}, {bi + 1})", ai
            except ValueError:
                pass
            # Variable end: for i = 1:N → range(N)
            try:
                ai = int(a_raw)
                if ai == 1:
                    return f"range({b})", 1
                else:
                    return f"range({ai}, {b} + 1)", ai
            except ValueError:
                pass
            return f"range(int({a}), int({b}) + 1)", None

        elif len(parts) == 3:
            a_raw, step_raw, b_raw = parts[0].strip(), parts[1].strip(), parts[2].strip()
            a    = self._expr(a_raw)
            step = self._expr(step_raw)
            b    = self._expr(b_raw)
            self.needs_numpy = True
            try:
                ai, si, bi = int(a_raw), int(step_raw), int(b_raw)
                if ai == 1 and si > 0:
                    # range(0, N, step) — adjust end
                    n = len(range(ai, bi + 1, si))
                    return f"range(0, {n}, {si})", 1   # offset by 1
                return f"range({ai}, {bi + si}, {si})", ai
            except ValueError:
                pass
            return f"np.arange({a}, {b} + {step}, {step})", None

        return self._expr(rng), None

    def _range(self, rng):
        """Legacy _range for non-for contexts (while conditions etc.)"""
        s, _ = self._range_for(rng)
        return s

    def _idx_slice_end(self, e):
        """Convert 1-based end-of-slice to Python exclusive upper bound.
        MATLAB a(1:3) → Python a[0:3], end value stays the same.
        """
        e = e.strip()
        if e in ("end", "-1", ""):
            return ""   # open slice a[x:]
        try:
            return str(int(e))
        except ValueError:
            pass
        # If contains a known 0-based loop var, no adjustment needed
        for var, start in self.loop_vars.items():
            if start == 1 and re.search(rf"\b{var}\b", e):
                return e
        return e

    def _idx_one(self, e):
        """Convert single 1-based index expression to 0-based Python.
        
        If e is a known loop variable that already starts from 0 (because
        we converted 'for i=1:N' to 'for i in range(N)'), no -1 needed.
        """
        e = e.strip()
        if e in ("end", "-1"):
            return "-1"
        # Integer constant: just subtract 1
        try:
            return str(int(e) - 1)
        except ValueError:
            pass
        # Pure variable that is a tracked loop var starting at MATLAB 1
        # → already 0-based in Python, no offset needed
        if e in self.loop_vars and self.loop_vars[e] == 1:
            return e
        # Simplify arithmetic patterns
        # (var+1)-1 = var  e.g. i+1 where i is 0-based loop var
        m = re.match(r"^(\w+)\s*\+\s*1$", e)
        if m and m.group(1) in self.loop_vars and self.loop_vars[m.group(1)] == 1:
            return f"{m.group(1)}+1"   # a(i+1) → a[i+1] (i already 0-based)
        m = re.match(r"^(\w+)\s*-\s*1$", e)
        if m and m.group(1) in self.loop_vars and self.loop_vars[m.group(1)] == 1:
            return f"{m.group(1)}-1"   # a(i-1) → a[i-1]
        # Generic expression with known loop var: add -1 only for unknown vars
        # Check if expression contains any known loop var
        for var, start in self.loop_vars.items():
            if start == 1 and re.search(rf"\b{var}\b", e):
                # Expression contains a 0-based loop var, no -1 needed
                return e
        # Default: subtract 1
        m = re.match(r"^(.+)\+\s*1$", e)
        if m:
            return m.group(1).strip()
        m = re.match(r"^(.+)-\s*1$", e)
        if m:
            return f"{m.group(1).strip()}-2"
        return f"{e}-1"

    def _index(self, s):
        """Convert MATLAB index args to Python 0-based."""
        parts = self._split_args(s)
        out = []
        for p in parts:
            p = p.strip()
            if p == ":":
                out.append(":")
            elif p.lower() == "end":
                out.append("-1")
            elif ":" in p:
                subs = p.split(":", 2)
                if len(subs) == 2:
                    a_raw = subs[0].strip()
                    b_raw = subs[1].strip()
                    # Convert start (1-based → 0-based)
                    if a_raw.lower() == "end":
                        a_py = "-1"
                    else:
                        a_py = self._idx_one(self._expr(a_raw))
                    # Convert end (stays same for exclusive Python slice)
                    if b_raw.lower() == "end" or b_raw == "":
                        b_py = ""
                    else:
                        b_e = self._expr(b_raw)
                        b_py = self._idx_slice_end(b_e)
                    out.append(f"{a_py}:{b_py}")
                else:
                    # a:step:b
                    a_raw  = subs[0].strip()
                    st_raw = subs[1].strip()
                    b_raw  = subs[2].strip()
                    a_py   = self._idx_one(self._expr(a_raw))
                    st_py  = self._expr(st_raw)
                    b_py   = ("" if b_raw.lower() == "end"
                              else self._idx_slice_end(self._expr(b_raw)))
                    out.append(f"{a_py}:{b_py}:{st_py}")
            else:
                # Single index — catch 'end' before calling _expr
                if p.lower() == "end":
                    out.append("-1")
                else:
                    out.append(self._idx_one(self._expr(p)))
        return ", ".join(out)
        return ", ".join(out)

    def _split_args(self, s):
        parts, depth, cur = [], 0, ""
        for ch in s:
            if ch == "(": depth += 1
            elif ch == ")": depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        if cur: parts.append(cur)
        return parts

    def _find_bracket_end(self, s, start):
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "[":   depth += 1
            elif s[i] == "]":
                depth -= 1
                if depth == 0: return i
        return -1

    def _is_variable_or_expr(self, s):
        """True if token looks like a variable/expression, not a plain scalar."""
        s = s.strip()
        # plain number → False
        try:
            float(s)
            return False
        except ValueError:
            pass
        return True

    def _matrix_to_numpy(self, inner):
        """Convert [1,2;3,4] inner → np.array(...)
           [a;b]  → np.vstack([a, b])
           [a,b]  → np.hstack([a, b])
           [a b]  → np.hstack([a, b])   (space-separated single row of variables)
        """
        self.needs_numpy = True
        rows = [r.strip() for r in re.split(r";", inner)]

        # Multi-row with variables → vstack
        if len(rows) > 1:
            row_elems = []
            for row in rows:
                elems = [e for e in re.split(r"[,\s]+", row.strip()) if e]
                row_elems.append(elems)
            # Check if any element is a variable (not scalar)
            all_elems = [e for row in row_elems for e in row]
            if any(self._is_variable_or_expr(e) for e in all_elems):
                # vstack: each row is one element (assume single variable per row)
                py_elems = [self._expr_no_matrix(row.strip()) for row in rows]
                return f"np.vstack([{', '.join(py_elems)}])"
            # All scalars → np.array([[...],[...]])
            py_rows = []
            for elems in row_elems:
                py_rows.append("[" + ", ".join(self._expr_no_matrix(e) for e in elems) + "]")
            return "np.array([" + ", ".join(py_rows) + "])"

        # Single row
        elems = [e for e in re.split(r"[,\s]+", rows[0].strip()) if e]
        if len(elems) > 1 and any(self._is_variable_or_expr(e) for e in elems):
            # hstack: multiple variables side by side
            py_elems = [self._expr_no_matrix(e) for e in elems]
            return f"np.hstack([{', '.join(py_elems)}])"
        # Single element or all scalars
        py_elems = [self._expr_no_matrix(e) for e in elems]
        return "np.array([" + ", ".join(py_elems) + "])"

    def _expr(self, expr):
        """Full expression conversion including matrix literals."""
        if not expr:
            return expr

        # Step 1: expand [...] matrix literals, replacing with np.array(...)
        # We do this by building the result character by character,
        # substituting [...] spans with placeholders first,
        # then processing the rest.
        if "[" in expr:
            chunks = []   # list of (text, is_already_converted)
            i = 0
            while i < len(expr):
                if expr[i] == "[":
                    end = self._find_bracket_end(expr, i)
                    if end != -1:
                        inner = expr[i+1:end].strip()
                        if not inner:
                            self.needs_numpy = True
                            chunks.append(("np.array([])", True))
                        else:
                            chunks.append((self._matrix_to_numpy(inner), True))
                        i = end + 1
                        continue
                if chunks and not chunks[-1][1]:
                    chunks[-1] = (chunks[-1][0] + expr[i], False)
                else:
                    chunks.append((expr[i], False))
                i += 1
            # Process non-converted chunks, leave converted ones alone
            result_parts = []
            for text, converted in chunks:
                if converted:
                    result_parts.append(text)
                else:
                    result_parts.append(self._expr_no_matrix(text))
            return "".join(result_parts)

        return self._expr_no_matrix(expr)

    def _expr_no_matrix(self, expr):
        """Expression conversion WITHOUT matrix literal handling."""
        if not expr:
            return expr

        # Strings
        expr = re.sub(r"'([^']*)'", r'"\1"', expr)

        # Operators
        expr = expr.replace(".*", " * ")
        expr = expr.replace("./", " / ")
        expr = expr.replace(".^", "**")
        expr = re.sub(r"(\w+)'", r"\1.T", expr)
        expr = re.sub(r"\^", "**", expr)
        expr = expr.replace("&&", " and ")
        expr = expr.replace("||", " or ")
        expr = re.sub(r"~=", "!=", expr)
        expr = re.sub(r"(?<![=!<>])~(?!=)", "not ", expr)
        expr = re.sub(r"\bend\b", "-1", expr)

        # Scalar constants
        for pat, repl in SCALAR_SUBS:
            if re.search(pat, expr):
                expr = re.sub(pat, repl, expr)
                if repl.startswith("np."): self.needs_numpy = True

        # Plot functions
        for mf, pf in PLOT_FUNCS.items():
            if re.search(rf"(?<!\.)\b{re.escape(mf)}\s*\(", expr):
                expr = re.sub(rf"(?<!\.)\b{re.escape(mf)}\s*\(", pf + "(", expr)
                self.needs_matplotlib = True

        # Math functions
        for mf, pf in MATH_FUNCS.items():
            pat = rf"(?<![.\w]){re.escape(mf)}\s*\("
            if re.search(pat, expr):
                expr = re.sub(pat, pf + "(", expr)
                if pf.startswith("np."): self.needs_numpy = True
                elif pf.startswith("scipy.interpolate"): self.needs_scipy_interp = True
                elif pf.startswith("scipy."): self.needs_scipy_io = True

        # Array indexing name(args) → name[args] for unknown variables
        def fix_idx(m):
            name, args = m.group(1), m.group(2)
            if name in ALL_KNOWN_FUNCS:
                return m.group(0)
            return f"{name}[{self._index(args)}]"

        expr = re.sub(r"\b([a-zA-Z_]\w*)\(([^()]+)\)", fix_idx, expr)

        return expr


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Octave/MATLAB to Python")
    parser.add_argument("input",  help="Input .m file")
    parser.add_argument("-o", "--output", help="Output .py file")
    args = parser.parse_args()
    result = OctaveConverter().convert_file(args.input)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Converted: {args.input} -> {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()