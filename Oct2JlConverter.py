#!/usr/bin/env python3
"""
oct2jl.py  —  Octave/MATLAB → Julia converter
Usage:
    python oct2jl.py input.m              # prints to stdout
    python oct2jl.py input.m -o out.jl   # writes file
"""

import re, argparse

# ─── mapping tables ───────────────────────────────────────────────────────────

# Functions that keep their name in Julia (no rename needed)
SAME_FUNCS = {
    "zeros","ones","rand","randn","reshape","diag","hcat","vcat",
    "inv","det","norm","svd","cross","dot","kron","pinv","rank",
    "sum","prod","cumsum","min","max","abs","real","imag","conj",
    "floor","ceil","round","mod","rem","sign","sort","unique",
    "sin","cos","tan","asin","acos","atan","sinh","cosh","tanh",
    "exp","log","log2","log10","sqrt","any","all","size","length",
    "println","print","string","isempty","push!","pop!","sparse",
    "fft","ifft","fftshift",
}

# Functions that need renaming
RENAME = {
    "linspace":   "range",
    "logspace":   "exp10.(range",
    "eye":        "I",
    "numel":      "length",
    "fliplr":     lambda args: f"reverse({args}, dims=2)",
    "flipud":     lambda args: f"reverse({args}, dims=1)",
    "repmat":     "repeat",
    "horzcat":    "hcat",
    "vertcat":    "vcat",
    "trace":      "tr",
    "eig":        "eigen",
    "eigs":       "eigs",
    "min":        "minimum",
    "max":        "maximum",
    "find":       "findall",
    "num2str":    "string",
    "str2num":    "parse(Float64, ",
    "str2double": "parse(Float64, ",
    "strcmp":     "==",
    "strsplit":   "split",
    "disp":       "println",
    "fprintf":    "@printf",
    "sprintf":    "@sprintf",
    "mean":       "mean",
    "std":        "std",
    "var":        "var",
    "median":     "median",
    "cov":        "cov",
    "corrcoef":   "cor",
    "atan2":      "atan",
    "meshgrid":   "meshgrid",
}

# Functions needing dot-broadcast in Julia
BROADCAST_FUNCS = {
    "sin","cos","tan","asin","acos","atan","sinh","cosh","tanh",
    "exp","log","log2","log10","sqrt","abs","real","imag","conj",
    "sign","floor","ceil","round",
}

# Pyplot mappings
PLOT_MAP = {
    "plot":      "plot",
    "plot3":     "plot3D",
    "semilogx":  "semilogx",
    "semilogy":  "semilogy",
    "loglog":    "loglog",
    "bar":       "bar",
    "hist":      "plt.hist",
    "histogram": "plt.hist",
    "imagesc":   "imshow",
    "pcolor":    "pcolormesh",
    "pcolormesh":"pcolormesh",
    "contour":   "contour",
    "contourf":  "contourf",
    "xlabel":    "xlabel",
    "ylabel":    "ylabel",
    "zlabel":    "zlabel",
    "title":     "title",
    "legend":    "legend",
    "colorbar":  "colorbar",
    "xlim":      "xlim",
    "ylim":      "ylim",
    "zlim":      "zlim",
    "subplot":   "subplot",
    "clf":       "clf",
    "cla":       "cla",
    "axis":      "axis",
    "clim":      "clim",
}

# scalar constant substitutions  (MATLAB → Julia)
CONST_MAP = [
    (r"\bpi\b",    "π"),
    (r"\bInf\b",   "Inf"),
    (r"\binf\b",   "Inf"),
    (r"\bNaN\b",   "NaN"),
    (r"\bnan\b",   "NaN"),
    (r"\btrue\b",  "true"),
    (r"\bfalse\b", "false"),
    (r"\beps\b",   "eps(Float64)"),
]

ALL_KNOWN = (SAME_FUNCS | set(RENAME.keys()) | set(PLOT_MAP.keys()) |
             {"scatter","figure","grid","hold","colorbar","close","drawnow",
              "range","I","tr","eigen","eigs","minimum","maximum","findall",
              "mean","std","var","median","cov","cor","repeat","println",
              "println","string","isempty","push!","pop!","parse",
              "hcat","vcat","zeros","ones","rand","randn","reshape",
              "diag","inv","det","norm","svd","cross","dot","kron",
              "pinv","rank","sum","prod","cumsum","sort","unique",
              "fft","ifft","fftshift","length","size","sparse",
             })


# ─── converter class ──────────────────────────────────────────────────────────

class Oct2Jl:
    def __init__(self):
        self.indent        = 0
        self.loop_vars     = {}   # varname → matlab_start (1 or other)
        self.use_LA        = False   # LinearAlgebra
        self.use_MAT       = False   # MAT.jl
        self.use_Printf    = False   # Printf
        self.use_Stats     = False   # Statistics
        self.use_PyPlot    = False   # PyPlot

    # ── public ────────────────────────────────────────────────────────────────

    def convert_file(self, path):
        with open(path, encoding="utf-8", errors="ignore") as f:
            return self.convert(f.read())

    def convert(self, src):
        self.indent = 0
        body_lines = []
        for raw in src.splitlines():
            out = self._process_line(raw)
            if out is not None:
                body_lines.append(out)
        body_lines = self._inject_nothing(body_lines)
        return "\n".join(self._header() + ["", ""] + body_lines)

    # ── header ────────────────────────────────────────────────────────────────

    def _header(self):
        h = ["# Generated by oct2jl.py"]
        if self.use_LA:     h.append("using LinearAlgebra")
        if self.use_MAT:    h.append("using MAT")
        if self.use_Printf: h.append("using Printf")
        if self.use_Stats:  h.append("using Statistics")
        if self.use_PyPlot: h.append("using PyPlot")
        return h

    # ── empty-block guard ─────────────────────────────────────────────────────

    def _inject_nothing(self, lines):
        """Insert 'nothing' where a block header has an empty body."""
        out = []
        for i, line in enumerate(lines):
            out.append(line)
            s = line.rstrip()
            # Julia block openers end with nothing (no colon like Python)
            if re.search(r"^\s*(for |while |if |elseif |else$|function )", s):
                cur_ind = len(s) - len(s.lstrip())
                nxt = next((lines[j] for j in range(i+1, len(lines))
                            if lines[j].strip()), None)
                if nxt is None or nxt.strip() == "end":
                    out.append(" " * (cur_ind + 4) + "nothing")
        return out

    # ── line processing ───────────────────────────────────────────────────────

    def _process_line(self, raw):
        line = raw.rstrip()
        if not line.strip():
            return ""

        stmts = self._split_semicolons(line.strip())
        results = []
        for s in stmts:
            code, cmt = self._split_comment(s.strip())
            code = code.strip()
            # strip trailing semicolon (Julia doesn't print by default anyway)
            code = code.rstrip(";").strip()
            if not code:
                if cmt:
                    results.append(self._ind() + "# " + cmt)
                continue
            r = self._stmt(code)
            if r is not None and r.strip():
                if cmt:
                    results.append(r + "  # " + cmt)
                else:
                    results.append(r)
        return "\n".join(results) if results else ""

    def _split_semicolons(self, line):
        """Split on ';' outside strings/parens/brackets."""
        parts, depth_p, depth_b, in_str, sc, cur = [], 0, 0, False, None, ""
        for ch in line:
            if not in_str and ch in ('"', "'"):
                in_str, sc = True, ch
            elif in_str and ch == sc:
                in_str = False
            elif not in_str:
                if   ch == "(": depth_p += 1
                elif ch == ")": depth_p -= 1
                elif ch == "[": depth_b += 1
                elif ch == "]": depth_b -= 1
                elif ch == ";" and depth_p == 0 and depth_b == 0:
                    parts.append(cur); cur = ""; continue
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

    def _ind(self, extra=0):
        return "    " * (self.indent + extra)

    # ── statement dispatcher ──────────────────────────────────────────────────

    def _stmt(self, code):
        ind = self._ind()

        # ── clear / clc ───────────────────────────────────────────────────────
        if re.match(r"^(clear|clc)\b", code):
            return ind + "# " + code

        # ── end (block closer) ────────────────────────────────────────────────
        if re.fullmatch(r"end", code):
            self.indent = max(0, self.indent - 1)
            return self._ind() + "end"

        # ── function definition ───────────────────────────────────────────────
        m = re.match(r"^function\s+(?:(\[?[\w\s,]+\]?)\s*=\s*)?(\w+)\s*\(([^)]*)\)", code)
        if m:
            ret, name, args = m.group(1), m.group(2), m.group(3).strip()
            comment = f"  # returns: {ret.strip().strip('[]')}" if ret else ""
            s = ind + f"function {name}({args}){comment}"
            self.indent += 1
            return s

        # ── return ────────────────────────────────────────────────────────────
        if code == "return":
            return ind + "return"

        # ── for loop ──────────────────────────────────────────────────────────
        m = re.match(r"^for\s+(\w+)\s*=\s*(.+)", code)
        if m:
            var, rng_raw = m.group(1), m.group(2).strip()
            rng_jl, start = self._for_range(rng_raw)
            self.loop_vars[var] = start
            s = ind + f"for {var} in {rng_jl}"
            self.indent += 1
            return s

        # ── while ─────────────────────────────────────────────────────────────
        m = re.match(r"^while\s+(.+)", code)
        if m:
            s = ind + f"while {self._expr(m.group(1).strip())}"
            self.indent += 1
            return s

        # ── if ────────────────────────────────────────────────────────────────
        m = re.match(r"^if\s+(.+)", code)
        if m:
            s = ind + f"if {self._expr(m.group(1).strip())}"
            self.indent += 1
            return s

        # ── elseif ────────────────────────────────────────────────────────────
        m = re.match(r"^elseif\s+(.+)", code)
        if m:
            self.indent = max(0, self.indent - 1)
            s = self._ind() + f"elseif {self._expr(m.group(1).strip())}"
            self.indent += 1
            return s

        # ── else ──────────────────────────────────────────────────────────────
        if code == "else":
            self.indent = max(0, self.indent - 1)
            s = self._ind() + "else"
            self.indent += 1
            return s

        if code == "break":    return ind + "break"
        if code == "continue": return ind + "continue"

        # ── save ──────────────────────────────────────────────────────────────
        m = re.match(r"^save\s*\(\s*['\"](.+?)['\"](.*)?\)", code)
        if m:
            fname = m.group(1)
            vars_ = [v.strip().strip("'\"") for v in (m.group(2) or "").split(",")
                     if v.strip().strip("'\"")]
            if fname.endswith(".mat"):
                self.use_MAT = True
                if vars_:
                    d = "Dict(" + ", ".join(f'"{v}" => {v}' for v in vars_) + ")"
                    return ind + f'matwrite("{fname}", {d})'
                return ind + f'# matwrite("{fname}", Dict(...))'
            else:
                delim = "','" if fname.endswith(".csv") else "' '"
                if len(vars_) == 1:
                    return ind + f'writedlm("{fname}", {vars_[0]}, {delim})'
                elif vars_:
                    return ind + f'writedlm("{fname}", hcat({", ".join(vars_)}), {delim})'
                return ind + f'# writedlm("{fname}", data)'

        # ── load ──────────────────────────────────────────────────────────────
        m = re.match(r"^load\s*\(?\s*['\"](.+?)['\"]", code)
        if m:
            fname = m.group(1)
            var   = re.sub(r"[^a-zA-Z0-9_]", "_", fname.rsplit(".", 1)[0])
            if fname.endswith(".mat"):
                self.use_MAT = True
                return ind + f'{var} = matread("{fname}")'
            elif fname.endswith(".csv"):
                return ind + f'{var} = readdlm("{fname}", \',\', Float64)'
            else:
                return ind + f'{var} = readdlm("{fname}")'

        # ── figure ────────────────────────────────────────────────────────────
        m = re.match(r"^figure\s*(\(.*\))?$", code)
        if m:
            self.use_PyPlot = True
            args = m.group(1)
            return ind + (f"figure({args.strip('()')})" if args else "figure()")

        # ── hold ──────────────────────────────────────────────────────────────
        m = re.match(r"^hold\s+(on|off)$", code)
        if m:
            return ""   # PyPlot holds by default; no-op

        # ── grid ──────────────────────────────────────────────────────────────
        m = re.match(r"^grid\s+(on|off)$", code)
        if m:
            self.use_PyPlot = True
            val = "true" if m.group(1) == "on" else "false"
            return ind + f"grid({val})"

        # ── drawnow / close all ───────────────────────────────────────────────
        if code in ("drawnow", "show"):
            self.use_PyPlot = True
            return ind + "plt.show()"
        if re.match(r"^close\s*(all)?$", code):
            self.use_PyPlot = True
            return ind + "plt.close(\"all\")"

        # ── colorbar ─────────────────────────────────────────────────────────
        if re.match(r"^colorbar\s*(\(\))?$", code):
            self.use_PyPlot = True
            return ind + "colorbar()"

        # ── axis ──────────────────────────────────────────────────────────────
        m = re.match(r"^axis\s+(equal|off|tight|on|image)$", code)
        if m:
            self.use_PyPlot = True
            return ind + f'axis("{m.group(1)}")'

        # ── shading ───────────────────────────────────────────────────────────
        m = re.match(r"^shading\s+(\w+)$", code)
        if m:
            return ind + f'# shading {m.group(1)}'

        # ── scatter ───────────────────────────────────────────────────────────
        m = re.match(r"^scatter\s*\((.+)\)$", code)
        if m:
            self.use_PyPlot = True
            args = [self._expr(a.strip()) for a in self._split_args(m.group(1))]
            args = [a for a in args if a not in ('"filled"', "'filled'")]
            if len(args) >= 4:
                return ind + f'scatter({args[0]}, {args[1]}, s={args[2]}, c={args[3]}, cmap="viridis")'
            elif len(args) == 3:
                return ind + f'scatter({args[0]}, {args[1]}, s={args[2]})'
            else:
                return ind + f'scatter({", ".join(args)})'

        # ── fprintf ───────────────────────────────────────────────────────────
        m = re.match(r"^fprintf\s*\((.+)\)$", code)
        if m:
            self.use_Printf = True
            return ind + f"@printf({m.group(1)})"

        # ── disp ──────────────────────────────────────────────────────────────
        m = re.match(r"^disp\s*\((.+)\)$", code)
        if m:
            return ind + f"println({self._expr(m.group(1).strip())})"

        # ── [V,D] = eig(A) ───────────────────────────────────────────────────
        m = re.match(r"^\[([^\]]+)\]\s*=\s*(eigs?)\s*\((.+)\)$", code)
        if m:
            self.use_LA = True
            lhs  = [v.strip() for v in m.group(1).split(",")]
            fn   = m.group(2)
            args = [self._expr(a.strip()) for a in self._split_args(m.group(3))]
            vec_var = lhs[0]
            val_var = lhs[1] if len(lhs) > 1 else "_D"
            astr = ", ".join(args)
            jfn  = "eigs" if fn == "eigs" else "eigen"
            return (ind + f"_ef = {jfn}({astr})\n" +
                    ind + f"{val_var} = Diagonal(real(_ef.values))\n" +
                    ind + f"{vec_var} = _ef.vectors")

        # ── [a,b,...] = expr ──────────────────────────────────────────────────
        m = re.match(r"^\[([^\]]+)\]\s*=\s*(.+)", code)
        if m:
            lhs = ", ".join(v.strip() for v in m.group(1).split(","))
            return ind + f"{lhs} = {self._expr(m.group(2).strip())}"

        # ── lhs = expr ────────────────────────────────────────────────────────
        m = re.match(r"^([\w.]+(?:\([^)=]+\))?)\s*=\s*(.+)", code)
        if m:
            lhs = self._lhs(m.group(1))
            rhs = self._expr(m.group(2).strip())
            return ind + f"{lhs} = {rhs}"

        # ── bare expression / function call ───────────────────────────────────
        return ind + self._expr(code)

    # ── for-range converter ───────────────────────────────────────────────────

    def _for_range(self, rng):
        """Return (julia_range_str, matlab_start).
        Julia is 1-based, so 'for i=1:N' stays 'for i in 1:N'.
        """
        parts = rng.split(":")
        if len(parts) == 2:
            a, b = parts[0].strip(), parts[1].strip()
            ja   = self._expr(a)
            jb   = self._expr(b)
            try:  start = int(a)
            except: start = None
            return f"{ja}:{jb}", start
        elif len(parts) == 3:
            a, step, b = parts[0].strip(), parts[1].strip(), parts[2].strip()
            ja, jstep, jb = self._expr(a), self._expr(step), self._expr(b)
            try:  start = int(a)
            except: start = None
            return f"{ja}:{jstep}:{jb}", start
        return self._expr(rng), None

    # ── indexing ──────────────────────────────────────────────────────────────

    def _index(self, s):
        """Convert MATLAB index expression to Julia.
        Julia is 1-based just like MATLAB — no offset needed!
        Just handle  :  and  end  keywords.
        """
        parts = self._split_args(s)
        out = []
        for p in parts:
            p = p.strip()
            if p == ":":
                out.append(":")
            elif p.lower() == "end":
                out.append("end")
            elif ":" in p:
                subs = p.split(":", 2)
                if len(subs) == 2:
                    a_raw, b_raw = subs[0].strip(), subs[1].strip()
                    a_jl = self._expr(a_raw)
                    b_jl = "end" if b_raw.lower() == "end" else self._expr(b_raw)
                    out.append(f"{a_jl}:{b_jl}")
                else:
                    a    = self._expr(subs[0].strip())
                    step = self._expr(subs[1].strip())
                    b    = "end" if subs[2].strip().lower() == "end" else self._expr(subs[2].strip())
                    out.append(f"{a}:{step}:{b}")
            else:
                out.append(self._expr(p))
        return ", ".join(out)

    def _lhs(self, s):
        """Convert LHS assignment target: a(i,j) → a[i,j]."""
        m = re.match(r"^(\w+)\((.+)\)$", s)
        if m and m.group(1) not in ALL_KNOWN:
            return f"{m.group(1)}[{self._index(m.group(2))}]"
        return s

    # ── expression converter ──────────────────────────────────────────────────

    def _expr(self, e):
        if not e:
            return e
        # First expand matrix literals [...] 
        if "[" in e:
            result, i = [], 0
            while i < len(e):
                if e[i] == "[":
                    end = self._bracket_end(e, i)
                    if end != -1:
                        inner = e[i+1:end].strip()
                        result.append(self._matrix(inner) if inner else "[]")
                        i = end + 1
                        continue
                result.append(e[i])
                i += 1
            e = "".join(result)
        return self._expr_core(e)

    def _expr_core(self, expr):
        if not expr:
            return expr

        # String literals: 'text' → "text"
        expr = re.sub(r"'([^']*)'", r'"\1"', expr)

        # Operators
        expr = expr.replace("~=", "!=")
        expr = re.sub(r"(?<![=!<>~])~(?!=)", "!", expr)
        expr = re.sub(r"\b&&\b", "&&", expr)
        expr = re.sub(r"\b\|\|\b", "||", expr)
        # Transpose: A.' → transpose(A),  A' stays A' (Julia same)
        expr = re.sub(r"(\w|\])\.'", lambda m: f"transpose({m.group(0)[:-2]})", expr)

        # Scalar constants
        for pat, repl in CONST_MAP:
            expr = re.sub(pat, repl, expr)

        # Imaginary: 2i, 3j → 2im, 3im  (but not bare i/j which are loop vars)
        expr = re.sub(r"(\d+(?:\.\d+)?)[ij]\b", r"\1im", expr)

        # plot functions: plot(...) → plot(...)  [PyPlot]
        for mname, jname in PLOT_MAP.items():
            pat = rf"(?<![.\w])\b{re.escape(mname)}\s*\("
            if re.search(pat, expr):
                self.use_PyPlot = True
                expr = re.sub(pat, jname + "(", expr)

        # Rename functions
        for mname, jval in RENAME.items():
            pat = rf"(?<![.\w])\b{re.escape(mname)}\s*\("
            if not re.search(pat, expr):
                continue
            if mname in ("mean","std","var","median","cov","corrcoef"):
                self.use_Stats = True
            if mname in ("eig","eigs"):
                self.use_LA = True
            if mname in ("fprintf","sprintf"):
                self.use_Printf = True
            if callable(jval):
                # lambda replacement: extract args and pass
                expr = re.sub(pat, lambda m, jv=jval: "__LAMBDA__", expr)
                # not perfect for lambdas; just rename for now
                expr = expr.replace("__LAMBDA__", jval.split("(")[0] + "(")
            else:
                expr = re.sub(pat, jval + "(", expr)

        # LinearAlgebra functions
        for fn in ("inv","det","norm","svd","cross","dot","kron","pinv","rank","tr","eigen","eigs","I"):
            if re.search(rf"(?<![.\w])\b{re.escape(fn)}\b", expr):
                self.use_LA = True
                break

        # Add dot-broadcast to math functions applied to expressions
        for fn in BROADCAST_FUNCS:
            jfn = RENAME.get(fn, fn)
            pat = rf"(?<![.\w])\b{re.escape(jfn)}\("
            if re.search(pat, expr):
                expr = re.sub(pat, jfn + ".(", expr)

        # Array indexing: name(args) → name[args]
        def replace_index(m):
            name = m.group(1)
            args = m.group(2)
            if name in ALL_KNOWN:
                return m.group(0)
            # Check it's not a function call by seeing if name is known
            return f"{name}[{self._index(args)}]"

        expr = re.sub(r"\b([a-zA-Z_]\w*)\(([^()]*)\)", replace_index, expr)

        return expr

    # ── matrix literal parser ──────────────────────────────────────────────────

    def _matrix(self, inner):
        """[1 2 3; 4 5 6] → [1 2 3; 4 5 6] (Julia same syntax!)
        But [a;b] (variable rows) → vcat(a, b)
        And [a,b] (variable cols) → hcat(a, b)
        """
        rows = re.split(r";", inner)
        rows = [r.strip() for r in rows if r.strip()]

        def is_number(s):
            try: float(s); return True
            except: return False

        def row_elems(r):
            return [e for e in re.split(r"[,\s]+", r.strip()) if e]

        if len(rows) > 1:
            all_e = [e for r in rows for e in row_elems(r)]
            # Variable rows → vcat
            if any(not is_number(e) for e in all_e):
                parts = [self._expr_core(r.strip()) for r in rows]
                return f"vcat({', '.join(parts)})"
            # Numeric matrix — Julia same syntax
            jrows = []
            for r in rows:
                elems = [self._expr_core(e) for e in row_elems(r)]
                jrows.append(" ".join(elems))
            return "[" + "; ".join(jrows) + "]"

        # Single row
        elems = row_elems(rows[0]) if rows else []
        if any(not is_number(e) for e in elems):
            parts = [self._expr_core(e) for e in elems]
            return f"hcat({', '.join(parts)})"
        # Numeric row vector
        parts = [self._expr_core(e) for e in elems]
        return "[" + " ".join(parts) + "]"

    # ── utilities ─────────────────────────────────────────────────────────────

    def _split_args(self, s):
        """Split on commas outside parens/brackets."""
        parts, depth, cur = [], 0, ""
        for ch in s:
            if ch in "([": depth += 1
            elif ch in ")]": depth -= 1
            if ch == "," and depth == 0:
                parts.append(cur); cur = ""
            else:
                cur += ch
        if cur: parts.append(cur)
        return parts

    def _bracket_end(self, s, start):
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "[":   depth += 1
            elif s[i] == "]":
                depth -= 1
                if depth == 0: return i
        return -1


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert Octave/MATLAB .m to Julia .jl")
    parser.add_argument("input",           help="Input .m file")
    parser.add_argument("-o","--output",   help="Output .jl file (default: stdout)")
    args = parser.parse_args()

    result = Oct2Jl().convert_file(args.input)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"Saved: {args.output}")
    else:
        print(result)

if __name__ == "__main__":
    main()