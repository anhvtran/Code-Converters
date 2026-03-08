# oct2py-julia — Octave/MATLAB Code Converter

Convert Octave/MATLAB `.m` scripts to **Python** (NumPy/Matplotlib) or **Julia** automatically.

```
oct2py_converter.py   →  converts .m  to  .py  (NumPy + Matplotlib)
oct2jl.py             →  converts .m  to  .jl  (Julia)
```

---

## Installation

No dependencies beyond Python 3.7+:

```bash
git clone https://github.com/yourname/oct2py-julia.git
cd oct2py-julia
```

To actually **run** the converted output you will need:

**Python output:**
```bash
pip install numpy matplotlib scipy
```

**Julia output:**
```julia
] add MAT PyPlot LinearAlgebra Statistics Printf
```

---

## Usage

```bash
# To Python
python oct2py_converter.py myscript.m              # print to stdout
python oct2py_converter.py myscript.m -o out.py    # write file

# To Julia
python oct2jl.py myscript.m                        # print to stdout
python oct2jl.py myscript.m -o out.jl              # write file
```

---

## What gets converted

### Syntax

| Octave | Python | Julia |
|--------|--------|-------|
| `% comment` | `# comment` | `# comment` |
| `'hello'` | `"hello"` | `"hello"` |
| `a(i,j)` | `a[i-1, j-1]` | `a[i, j]` |
| `a(end, :)` | `a[-1, :]` | `a[end, :]` |
| `a(1:3, :)` | `a[0:3, :]` | `a[1:3, :]` |
| `a(2:end, :)` | `a[1:, :]` | `a[2:end, :]` |
| `A'` | `A.T` | `A'` |
| `A.'` | `A.T` | `transpose(A)` |
| `A .* B` | `A * B` | `A .* B` |
| `A .^ 2` | `A ** 2` | `A .^ 2` |
| `pi`, `inf`, `NaN` | `np.pi`, `np.inf`, `np.nan` | `π`, `Inf`, `NaN` |

### Matrix literals

| Octave | Python | Julia |
|--------|--------|-------|
| `[1 2 3; 4 5 6]` | `np.array([[1,2,3],[4,5,6]])` | `[1 2 3; 4 5 6]` |
| `[a; b]` | `np.vstack([a, b])` | `vcat(a, b)` |
| `[a, b]` | `np.hstack([a, b])` | `hcat(a, b)` |

### Control flow

| Octave | Python | Julia |
|--------|--------|-------|
| `for i = 1:N` | `for i in range(N):` | `for i in 1:N` |
| `for i = 1:2:10` | `for i in range(0,5,2):` | `for i in 1:2:10` |
| `while cond` | `while cond:` | `while cond` |
| `if/elseif/else/end` | `if/elif/else:` | `if/elseif/else/end` |
| `function y=f(x)` | `def f(x):` | `function f(x)` |

> **Index note:** Python is 0-based so indices are shifted.  
> `for i = 1:N` → `for i in range(N)` with `a[i]` (not `a[i-1]`).  
> Julia is 1-based like MATLAB so no shift needed.

### Math functions

| Octave | Python | Julia |
|--------|--------|-------|
| `linspace(a,b,n)` | `np.linspace(a,b,n)` | `range(a,b,n)` |
| `zeros(m,n)` | `np.zeros((m,n))` | `zeros(m,n)` |
| `eye(n)` | `np.eye(n)` | `I` |
| `sin(x)`, `exp(x)` | `np.sin(x)`, `np.exp(x)` | `sin.(x)`, `exp.(x)` |
| `[V,D]=eig(A)` | `D_v,V=np.linalg.eig(A)` then `D=np.diag(D_v)` | `_ef=eigen(A); D=Diagonal(_ef.values); V=_ef.vectors` |
| `[V,D]=eig(A,B)` | `scipy.linalg.eig(A,B)` | `eigen(A,B)` |

### File I/O

| Octave | Python | Julia |
|--------|--------|-------|
| `save('f.mat','a','b')` | `scipy.io.savemat('f.mat', {"a":a,"b":b})` | `matwrite("f.mat", Dict("a"=>a,"b"=>b))` |
| `load('f.mat')` | `f = scipy.io.loadmat('f.mat')` | `f = matread("f.mat")` |
| `save('f.txt','a')` | `np.savetxt('f.txt', a)` | `writedlm("f.txt", a, ' ')` |
| `load('f.txt')` | `np.loadtxt('f.txt')` | `readdlm("f.txt")` |

### Plotting

Both converters target **PyPlot** (Matplotlib interface).

| Octave | Python | Julia |
|--------|--------|-------|
| `figure(1)` | `plt.figure(1)` | `figure(1)` |
| `plot(x,y,'r-')` | `plt.plot(x,y,"r-")` | `plot(x,y,"r-")` |
| `hold on` | *(no-op, Matplotlib holds by default)* | *(no-op)* |
| `scatter(x,y,sz,c,'filled')` | `plt.scatter(x,y,s=sz,c=c,cmap='viridis')` | `scatter(x,y,s=sz,c=c,cmap="viridis")` |
| `xlabel('...')` | `plt.xlabel('...')` | `xlabel("...")` |
| `grid on` | `plt.grid(True)` | `grid(true)` |
| `colorbar` | `plt.colorbar()` | `colorbar()` |

---

## Example

**Input** (`example.m`):
```matlab
clear all

a = [1,2,3; 4,5,6];
x = linspace(0, 2*pi, 100);
y = sin(x) .* cos(x);

for i = 1:3
    row = a(i, :);
end

figure
plot(x, y, 'r-')
grid on
save('data.mat', 'a')
```

**Python output:**
```python
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

a = np.array([[1, 2, 3], [4, 5, 6]])
x = np.linspace(0, 2*np.pi, 100)
y = np.sin(x) * np.cos(x)

for i in range(3):
    row = a[i, :]

plt.figure()
plt.plot(x, y, "r-")
plt.grid(True)
scipy.io.savemat('data.mat', {"a": a})
```

**Julia output:**
```julia
using LinearAlgebra
using MAT
using PyPlot

a = [1 2 3; 4 5 6]
x = range(0, 2*π, 100)
y = sin.(x) .* cos.(x)

for i in 1:3
    row = a[i, :]
end

figure()
plot(x, y, "r-")
grid(true)
matwrite("data.mat", Dict("a" => a))
```

---


## License

MIT


