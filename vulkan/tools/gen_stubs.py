#!/usr/bin/env python3
"""gen_stubs.py LINK_LOG OUT.cpp [--undname=EXE] : stubs for every symbol MSVC's linker could not find.

LINK_LOG is the raw output of a failed `cmake --build build-vulkan --config Release`.  link.exe names every symbol it
cannot resolve, on one line, twice over - the readable form in quotes and its own decoration in parentheses:

    strata_vulkan_host.lib(native_dense.obj) : error LNK2019: unresolved external symbol
      "void __cdecl strata::kernels::native_q8_0_packed_unregister(void const *)"
      (?native_q8_0_packed_unregister@kernels@strata@@YAXPEBX@Z) referenced in function "public: __cdecl
      strata::core::NativeDense::~NativeDense(void)" (...)

Those decorations are MSVC's own (`?name@ns@ns@@<cc><types>@Z`), not Itanium, so a python Itanium demangler cannot read
them; undname.exe from the installed MSVC toolset is the demangler this tool uses, in batch: the decorations go into a
temp file, one per line, and `undname <file>` answers with one demangled signature per line in the same order.

Each demangled signature is turned back into an out-of-line definition, in the declaring namespace (free functions) or
out of line (members, ctors, dtors), with the header that declares it pulled in so the definition has to match the
declaration the engine compiled against - which is what makes cl.exe emit the decoration the linker asked for.  Every
body is `strata::vulkan_engine::missing("<qualified name>")` (vulkan/src/engine/missing.cpp): print the name, exit
non-zero.  A stub is a stop with a label, never a result.

Entry points that already have a Vulkan implementation are defined by the runtime, so they are not unresolved and get
no stub.  The ones this port defines itself are named in IMPLEMENTED below: a link log is evidence about one build, and
a log taken before a kernel was ported still names it, so the list is what keeps a regeneration from putting a stub back
over a real body - a duplicate definition cl.exe would reject at the next link.  Add the name here in the same step as
the file that implements it (and take the stub out of vulkan/src/engine/stubs.cpp).  Regenerate - do not patch - from a
fresh link log after any change to the host tree or after porting one:

    python vulkan/tools/gen_stubs.py build-vulkan/logs/link.log vulkan/src/engine/stubs.cpp

ONE NAME SPACE IS REFUSED, on purpose: `strata::gpu::*`.  That is this port's own runtime face, declared by
vulkan/include/strata/gpu/runtime.hpp and forwarded to by every name vulkan/include/vulkan_compat/cuda_runtime.h maps.
An unresolved name there is not a kernel that has not been ported, it is the runtime's own API missing a definition, and
a stub whose body prints a name and exits 3 is the right answer for a kernel entry point and a destructive one for a
runtime call - `cudaGetLastError()` is not a kernel, and stubbing it turns an error report into a process death.  It is
also how a hole stops being visible: S4's link had 53 unresolved `strata::gpu::*` symbols and the generated stubs made
the link close over them.  This tool now names them and refuses; the fix belongs in vulkan/src/gpu/runtime.cpp.
"""

import glob
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))          # <repo>/vulkan/tools
VULKAN = os.path.dirname(HERE)                              # <repo>/vulkan
REPO = os.path.dirname(VULKAN)                              # <repo>

# link.exe's own wording.  LNK2019 carries the referencing function, LNK2001 does not; both name the symbol the same way.
UNRESOLVED_RE = re.compile(r'unresolved external symbol "((?:[^"\\]|\\.)*)"\s+\((\?[^)]*)\)')
REFERENCED_RE = re.compile(r'referenced in function "((?:[^"\\]|\\.)*)"\s+\((\?[^)]*)\)')

# The headers the engine's own TUs include, and so the ones a stub may include to see a declaration.
HDR_DIRS = ["include/strata/kernels", "include/strata/prefill", "include/strata/core", "include/strata/platform",
            "include/strata/ngram", "include/strata/spec", "include/strata/program", "include/strata/qwen35",
            "include/strata/artifact", "include/strata/plan", "vulkan/include/strata/gpu"]

# Entry points this port already IMPLEMENTS, keyed by the qualified name as undname spells it, valued by the file that
# defines them.  A link log is evidence about one build: a log captured before a kernel was ported still names it, and a
# stub written over the real body is a duplicate definition cl.exe refuses.  Porting one means adding its name here in
# the same step as the file that answers it, and taking its stub out of vulkan/src/engine/stubs.cpp (or regenerating
# that file from a link log that no longer reports the symbol).  add_inplace is the first entry on this list: it was
# the first unresolved symbol this build ever reported, out of strata_vulkan_host.lib(expert_source.obj), and
# vulkan/src/kernels/elementwise.cpp is what answers it now - which is why vulkan/probe/stub_probe.cpp calls
# strata::kernels::scale_inplace instead: the probe's contract is to land inside a stub, and scale_inplace is still one.
IMPLEMENTED = {
    "strata::kernels::add_inplace": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::scale_inplace": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::silu_inplace": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::gdn_gate": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::rms_norm_weighted": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::embedding_gather": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::f32_to_f16_bulk": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::f32_to_bf16_bulk": "vulkan/src/kernels/elementwise.cpp",
    "strata::kernels::dequant_bf16": "vulkan/src/kernels/dequant_bf16.cpp",
}

ACCESS = ("public:", "protected:", "private:")
CC = ("__cdecl", "__thiscall", "__stdcall", "__fastcall", "__vectorcall")

# The tool compiles every candidate header the way the engine target compiles its TUs, and skips the ones that do not
# stand on their own.  These are vulkan/CMakeLists.txt:204-212's include directories and defines - spelled RELATIVE
# to the repo, because cl.exe is run with the repo as its working directory: measured on this PC, an absolute /I whose
# path contains a space (this repo lives under `C:\Users\Pure Fox/...`) is dropped by a cl.exe run that is given more
# than one source file, and every probe then answers C1083 for headers that compile perfectly on their own.
CL_INCS = ["/Ivulkan/include/vulkan_compat", "/Ivulkan/include", "/Iinclude", "/Isrc/core", "/Ithird_party/ggml"]
CL_BASE = ["/nologo", "/std:c++20", "/EHsc", "/utf-8", "/W1", "/c", "/MP", "/DSTRATA_NATIVE_EXPERTS=1",
           '/DSTRATA_VERSION="0.1.44.1-vulkan"']
SCRATCH = "build-vulkan/logs/hdrprobe"                      # the probe TUs and their .obj files stay here


def include_spelling(h):
    """The path as the compiler sees it: /I<repo>/include and /I<vulkan>/include are on the search path."""
    for prefix in ("include/", "vulkan/include/"):
        if h.startswith(prefix):
            return h[len(prefix):]
    return h


def find_undname(explicit):
    if explicit:
        return explicit
    hits = sorted(p for pat in [r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\undname.exe",
                                r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\HostX86\x64\undname.exe"]
                  for p in glob.glob(pat))
    if hits:
        return hits[-1]                                          # the newest installed toolset
    for d in os.environ.get("PATH", "").split(os.pathsep):
        f = os.path.join(d, "undname.exe")
        if os.path.isfile(f):
            return f
    sys.exit("no undname.exe found: pass --undname=<path to the MSVC toolset's undname.exe>")


def find_cl():
    """cl.exe (the 64-bit host, 64-bit target one) by absolute path, plus the INCLUDE list vcvars would have set, so
    no shell round-trip is needed.  Getting this wrong is loud on purpose: a bad INCLUDE list looks exactly like a
    header that does not compile."""
    tools = sorted(glob.glob(r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\bin\Hostx64\x64\cl.exe"))
    if not tools:
        return None, ""
    cl = tools[-1]
    # cl.exe lives in <toolset>/bin/Hostx64/x64 and its own C++ headers in <toolset>/include: read the second from the
    # same glob pattern as the first, so the two cannot drift apart.
    toolset = sorted(os.path.dirname(p) for p in glob.glob(
        r"C:\Program Files*\Microsoft Visual Studio\*\*\VC\Tools\MSVC\*\include\cstdio"))
    if not toolset:
        sys.exit("found %s but not its own include directory" % cl)
    inc = [toolset[-1]]
    for d in sorted(glob.glob(r"C:\Program Files*\Windows Kits\10\Include\*")):
        for sub in ("ucrt", "um", "shared", "winrt", "cppwinrt"):
            p = os.path.join(d, sub)
            if os.path.isdir(p):
                inc.append(p)
    return cl, os.pathsep.join(inc)


# ------------------------------------------------------------------ 1. the unresolved symbols, from the link report

def read_log(path):
    """The link log's bytes, decoded whatever PowerShell wrote them as.

    This matters because it fails silently in the wrong direction.  A transcript written by PowerShell's Tee-Object
    starts with the UTF-16LE BOM (measured here: b'\\xff\\xfe-\\x00-\\x00 \\x00S\\x00e\\x00l\\x00e\\x00c\\x00t\\x00i\\x00o\\x00n\\x00'
    for build-vulkan/logs/s2gate-00.log, against b'\\xef\\xbb\\xbf' for the older s4-iter-02.log).  Opened as the
    default text encoding, every character of a UTF-16 log arrives with a NUL between it, UNRESOLVED_RE matches
    nothing, and the tool reports "names no unresolved external symbol: nothing to stub" - a tool failure that looks
    exactly like a build with a clean link.
    """
    data = open(path, "rb").read()
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        text = data.decode("utf-16", errors="replace")
    elif data[:3] == b"\xef\xbb\xbf":
        text = data.decode("utf-8-sig", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")
    return text


def parse_log(path):
    """Every unresolved external symbol link.exe named, deduplicated by decoration, with the object that asked for it
    and the error number.  Read line by line: one diagnostic is one line, and a demangled name is often long enough to
    push the `foo.obj : error LNK2019:` prefix a hundred characters away from the symbol it belongs to."""
    out = {}
    for line in read_log(path).splitlines():
        # MSBuild writes the requester as `foo.obj :` or `somelib.lib(bar.obj) :`; both are kept whole so a stub's
        # comment names the object that asked for the symbol, exactly as link.exe named it.
        source = re.search(r'([^\s:]+\.(?:obj|lib)(?:\([^\s()]+\))?)\s*:\s*error (LNK\d+)', line)
        for m in UNRESOLVED_RE.finditer(line):
            demangled, mangled = m.group(1), m.group(2)
            if mangled in out:
                out[mangled]["mentions"] += 1
                continue
            ref = REFERENCED_RE.search(line, m.end())
            out[mangled] = {"mangled": mangled, "log_demangled": demangled, "mentions": 1,
                            "obj": source.group(1) if source else "?", "code": source.group(2) if source else "?",
                            "ref": ref.group(1) if ref else ""}
    return out


# ------------------------------------------------------------------ 2. demangling: undname.exe, one batch call

def demangle(names, undname):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    tmp.write("\n".join(names) + "\n")
    tmp.close()
    r = subprocess.run([undname, tmp.name], capture_output=True, text=True)
    os.unlink(tmp.name)
    lines = [l for l in r.stdout.splitlines() if l.strip()]
    if len(lines) != len(names):
        sys.exit("undname returned %d line(s) for %d decoration(s)" % (len(lines), len(names)))
    return lines


# ------------------------------------------------------------------ 3. demangled signature -> definition

def tidy(demangled):
    s = re.sub(r"\b__ptr64\b|\b__ptr32\b", " ", demangled)
    return re.sub(r"\s+", " ", s).strip()


def strip_keywords(s):
    """undname spells types with elaborated type specifiers (`class std::vector<...>`, `struct X`, `enum E`).  That is
    its description of the type, not source spelling: `class std::vector<int, struct std::allocator<int> >` is not a
    valid template-id in C++.  The decoration depends on the type, not on the keyword, so they come out."""
    return re.sub(r"\b(?:class|struct|union|enum)\s+", "", s)


QUAL_RE = re.compile(r"(?:(?:~?[A-Za-z_][A-Za-z0-9_]*::)*~[A-Za-z_][A-Za-z0-9_]*"
                     r"|(?:[A-Za-z_][A-Za-z0-9_]*::)*[A-Za-z_][A-Za-z0-9_]*|operator\S*)$")


def parse_sig(demangled):
    """'public: bool __cdecl strata::core::X::f(int)const' -> kind / qual / ret / params / tail"""
    s = tidy(demangled)
    for a in ACCESS:
        if s.startswith(a):
            s = s[len(a):].strip()
    j = s.rfind(")")
    depth, open_at = 0, -1
    for i in range(j, -1, -1):
        if s[i] == ")":
            depth += 1
        elif s[i] == "(":
            depth -= 1
            if depth == 0:
                open_at = i
                break
    if open_at < 0:
        return None
    head, params, tail = s[:open_at].strip(), s[open_at + 1:j].strip(), s[j + 1:].strip()
    m = QUAL_RE.search(head)
    if not m:
        return None
    qual = m.group(0)
    # `static` comes out too: undname prints it for a static member (`public: static unsigned __int64 __cdecl
    # strata::core::RemoteExpertOpt::metadata_bytes(void)`), and repeating it on an out-of-line definition is C2724,
    # "'static' should not be used on member functions defined at file scope".  The linkage is the class's business.
    ret = re.sub(r"\s+", " ", re.sub(r"\b(?:%s|static)\b" % "|".join(CC), " ", head[:m.start()])).strip()
    parts = qual.split("::")
    last = parts[-1]
    kind = "free"
    if last.startswith("~"):
        kind, ret = "dtor", ""
    elif len(parts) >= 2 and last == parts[-2]:
        kind, ret = "ctor", ""
    if params == "void":
        params = ""
    return {"kind": kind, "qual": qual, "ret": strip_keywords(ret), "params": strip_keywords(params),
            "tail": tail, "parts": parts}


# ------------------------------------------------------------------ 4. which header declares it

CACHE = {}


def list_headers():
    """Every candidate header, as a path relative to the repo, so the file this tool writes stays portable."""
    out = []
    for d in HDR_DIRS:
        out += sorted(os.path.relpath(p, REPO).replace("\\", "/")
                      for p in glob.glob(os.path.join(REPO, d) + "/*.hpp"))
    for h in out:
        CACHE[h] = open(os.path.join(REPO, h), errors="replace").read()
    return out


def probe_headers(hdrs, cl):
    """One cl run over one-include TUs: the headers that do not compile on their own cannot be included by a stub.
    cl names the source file on every diagnostic, so each error line is attributed to the header it came from.  Every
    path handed to cl here is relative to the repo and cl runs with the repo as its working directory - see CL_INCS."""
    if not cl:
        return hdrs, []
    scratch_abs = os.path.join(REPO, SCRATCH)
    os.makedirs(scratch_abs, exist_ok=True)
    files, probes = {}, {}
    for i, h in enumerate(hdrs):
        rel = SCRATCH + "/h%04d.cpp" % i
        open(os.path.join(REPO, rel), "w").write('#include "%s"\n' % include_spelling(h))
        files["h%04d.cpp" % i] = h
        probes[rel] = os.path.join(scratch_abs, "h%04d.obj" % i)
    r = subprocess.run([cl] + CL_BASE + CL_INCS + ["/Fo" + SCRATCH + os.sep] + list(probes),
                       capture_output=True, text=True, cwd=REPO)
    # A cl run that produced no objects compiled nothing, and "nothing failed" would then read as "all good": the
    # usual cause is an INCLUDE list that does not point at the toolset, which answers C1083 for <cstdio>.
    if not any(os.path.isfile(o) for o in probes.values()):
        sys.exit("the header probe compiled nothing (cl exit %s). Its output:\n%s" % (r.returncode, r.stdout[-3000:]))
    bad = set()
    for line in (r.stdout + r.stderr).splitlines():
        if "error " not in line and "fatal " not in line:
            continue
        m = re.search(r'compiling source file (.+?)(?:\.cpp)', line)
        name = os.path.basename(m.group(1) + ".cpp") if m else None
        if name is None:
            m2 = re.search(r"\b(h\d{4}\.cpp)", line)
            name = m2.group(1) if m2 else None
        if name in files:
            bad.add(files[name])
    for f in list(probes) + list(files):
        for p in (os.path.join(REPO, f), os.path.join(scratch_abs, os.path.basename(f)[:-4] + ".obj")):
            try:
                os.unlink(p)
            except OSError:
                pass
    return [h for h in hdrs if h not in bad], sorted(bad)


def code_only(text):
    """A header with its comments blanked out.  Half of these headers *discuss* each other's kernels in prose
    (`mrope.hpp:31` talks about `build_rope_table`, which `rope.hpp:47,53` declares), and matching on the raw text
    picks a header that only mentions the name - then the declaration is not visible, and C++ will not let a file
    introduce a new member of a namespace it can see: cl.exe answers C2039."""
    t = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", t)


def declarations(text, name):
    """Every `NAME(...)` declaration in this header's *code*: the span from the name to the `;` that ends it, kept only
    when that span closes before any `{` (a declaration, not a definition - a definition is not unresolved).  `tail` is
    what follows the parameter list: `const`, `noexcept`, both, or nothing."""
    out = []
    for m in re.finditer(r"\b" + re.escape(name) + r"\s*\(", text):
        i, depth = m.end() - 1, 0
        while i < len(text):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        else:
            continue
        j = i
        while j < len(text) and text[j] not in ";{":
            j += 1
        if j >= len(text) or text[j] == "{":
            continue
        out.append({"span": text[m.start():j + 1], "params": text[m.end():i], "tail": text[i + 1:j]})
    return out


def arity(params):
    """Top-level parameter count, counting through nested () and <>."""
    if not params.strip() or params.strip() == "void":
        return 0
    depth, n = 0, 1
    for ch in params:
        if ch in "(<":
            depth += 1
        elif ch in ")>":
            depth -= 1
        elif ch == "," and depth == 0:
            n += 1
    return n


def declaring_header(sig, hdrs, texts):
    """The header whose declaration this definition must match.  A candidate has to declare the symbol in code, with the
    same parameter count and the same const-ness as the signature link.exe named; ties go to the header closest to the
    symbol's own namespace.  Returns (header, declaration, is_member): the declaration is where the two things the
    decoration does not carry - `static` and `noexcept` - are read from, and is_member says the symbol sits in a class
    some header declares, which is the case that cannot be defined out of line without that header."""
    parts, last = sig["parts"], sig["parts"][-1]
    name = last[1:] if last.startswith("~") else last
    cls = parts[-2] if len(parts) >= 2 else None
    want_ns, want_arity = parts[:-1], arity(sig["params"])
    want_const = "const" in sig["tail"]
    is_member = False
    best, best_key, best_decl = None, None, None
    for h in hdrs:
        code = code_only(texts[h])
        decls = declarations(code, name)
        if not decls:
            continue
        sees_class = bool(cls) and bool(re.search(r"\bclass\s+" + re.escape(cls) + r"\b", code))
        if cls is not None and sees_class:
            is_member = True
        if sig["kind"] in ("ctor", "dtor") and not sees_class:
            continue
        matching = [d for d in decls if arity(d["params"]) == want_arity
                    and ("const" in d["tail"]) == want_const]
        if not matching:
            continue
        rank = 0
        seg = h.split("/")
        for i, ns in enumerate(reversed(want_ns)):
            if ns not in seg:
                rank += 100 - i
        key = (0 if sees_class else 1, rank)
        if best_key is None or key < best_key:
            best, best_key, best_decl = h, key, matching[0]
    return best, best_decl, is_member


# ------------------------------------------------------------------ 5. emit the stub file

def stub_head(sig):
    tail = (" " + sig["tail"]) if sig["tail"] else ""
    if not sig["ret"]:
        return "%s(%s)%s" % (sig["qual"], sig["params"], tail)
    return "%s %s(%s)%s" % (sig["ret"], sig["qual"], sig["params"], tail)


def stub_body(sig, qual):
    r = sig["ret"]
    out = 'strata::vulkan_engine::missing("%s");' % qual
    if not r or r == "void":
        return out
    if r.endswith("&"):
        return out + " static std::remove_cv_t<std::remove_reference_t<%s>> s{}; return s;" % r
    return out + " return {};"


def main():
    positional = [a for a in sys.argv[1:] if not a.startswith("--")]
    undname_arg = None
    for a in sys.argv[1:]:
        if a.startswith("--undname="):
            undname_arg = a.split("=", 1)[1]
    if len(positional) < 2:
        sys.exit(__doc__)
    log_path, out_path = positional[0], positional[1]

    # The rule this phase writes down: generated stubs live under vulkan/ (or the build tree), never in the shared
    # host tree.  Refuse rather than leave a file in ../src.
    out_abs = os.path.abspath(out_path)
    src = os.path.join(REPO, "src") + os.sep
    if out_abs.startswith(src):
        sys.exit("refusing to write %s: generated stubs go under vulkan/, never under %s" % (out_path, src))

    undname = find_undname(undname_arg)
    cl, include_env = find_cl()
    if cl:
        os.environ["INCLUDE"] = include_env
        os.environ["PATH"] = os.path.dirname(cl) + os.pathsep + os.environ.get("PATH", "")

    unresolved = parse_log(log_path)
    if not unresolved:
        sys.exit("%s names no unresolved external symbol: nothing to stub" % log_path)
    names = sorted(unresolved)
    demangled = demangle(names, undname)

    hdrs = list_headers()
    texts = CACHE                                          # each header's text, read once by list_headers()
    good, skipped = probe_headers(hdrs, cl)

    defs, unmatched, used = [], [], set()
    runtime_face = []                                       # unresolved names in strata::gpu:: - refused, not stubbed
    implemented = []                                        # names on IMPLEMENTED - real body, no stub written
    histogram = {}
    for mangled, dem in zip(names, demangled):
        sig = parse_sig(dem)
        if not sig:
            unmatched.append((mangled, dem, "no signature could be read from it"))
            continue
        ns = "::".join(sig["parts"][:-1])
        histogram[ns] = histogram.get(ns, 0) + 1
        if ns == "strata::gpu" or ns.startswith("strata::gpu::"):
            runtime_face.append((mangled, dem))
            continue
        if sig["qual"] in IMPLEMENTED:
            implemented.append((sig["qual"], IMPLEMENTED[sig["qual"]], mangled, unresolved[mangled]))
            continue
        h, decl, is_member = declaring_header(sig, good, texts)
        if not h:
            where = declaring_header(sig, hdrs, texts)[0]
            unmatched.append((mangled, dem, "declared by %s" % (where or "no header this tool can see")
                              + (", which the compile probe rejected" if where else "")))
            continue
        # `noexcept` is not in the decoration (MSVC mangles a noexcept function exactly like one that is not), but the
        # declaration has it, and a definition that drops it is C2382 "redefinition; different exception specifications".
        if "noexcept" in decl["tail"] and "noexcept" not in sig["tail"]:
            sig["tail"] = (sig["tail"] + " noexcept").strip()
        used.add(h)
        defs.append((sig["qual"], stub_head(sig), stub_body(sig, sig["qual"]), mangled, dem,
                     unresolved[mangled], h))

    if runtime_face:
        print("undname (demangling route) : %s" % undname)
        print("link log %s" % log_path)
        print("REFUSED   : %d unresolved symbol(s) in strata::gpu:: - this port's own runtime face, not a kernel"
              % len(runtime_face))
        print("            entry point.  A stub that prints a name and exits 3 is right for a kernel and turns a")
        print("            runtime call (cudaGetLastError, cudaSetDevice, ...) into a process death, so it is not")
        print("            generated here.  Nothing was written.")
        for m, d in runtime_face:
            print("  REFUSED %s" % d)
        print("Fix: define them in vulkan/src/gpu/runtime.cpp, then re-run this tool on a fresh link log.")
        sys.exit(2)

    defs.sort(key=lambda d: (d[0], d[1]))

    print("undname (demangling route) : %s" % undname)
    print("cl (header probe)          : %s" % (cl or "none"))
    print("link log %s" % log_path)
    # read_log(), not open(): the mention count is computed from the SAME decoded text parse_log() saw.  A plain
    # open() on a PowerShell-written UTF-16 log reads each character as two bytes and UNRESOLVED_RE matches none of
    # them, so the summary line printed "unresolved mentions : 0" for a log that had just produced 280 stubs.
    print("  unresolved mentions      : %d   distinct decorations: %d" % (len(UNRESOLVED_RE.findall(
        read_log(log_path))), len(names)))
    print("  first decoration read    : %s" % names[0])
    print("  undname says             : %s" % demangled[0])
    print("headers probed  : %d  standalone-compilable: %d  skipped: %d" % (len(hdrs), len(good), len(skipped)))
    for h in skipped:
        print("  HEADER-SKIPPED %s" % h)
    print("stubs written   : %d" % len(defs))
    if implemented:
        print("  IMPLEMENTED   : %d name(s) this port already defines; no stub written for them" % len(implemented))
        for qual, where, mangled, info in implemented:
            print("    %-56s %s" % (qual, where))
            print("      %s for %s, reported by %s (%d mention(s)) - a body exists, so the stub does not"
                  % (info["code"], mangled, info["obj"], info["mentions"]))
    for ns in sorted(histogram, key=lambda k: (-histogram[k], k)):
        print("  %-42s %d" % (ns, histogram[ns]))
    for m, d, why in unmatched:
        print("  UNMATCHED %s  (%s)  %s" % (m, d, why))

    head = [
        "// vulkan/src/engine/stubs.cpp - GENERATED by vulkan/tools/gen_stubs.py: do not edit by hand.",
        "//",
        "// Source of this list: the link log %s.  Every entry below is a symbol link.exe reported as"
        % os.path.relpath(log_path, REPO).replace("\\", "/"),
        "// unresolved - MSVC named it, undname.exe decoded it, the tool rebuilt the definition from that signature.",
        "// Nothing here is implemented: each body prints the qualified name through strata::vulkan_engine::missing()",
        "// (vulkan/src/engine/missing.cpp) and exits non-zero, so the engine stops at an entry point that has no",
        "// Vulkan implementation instead of running on something that was never written.",
        "//",
        "// Regenerate from a fresh link log after any change to the host tree, or after porting one of these:",
        "//   python vulkan/tools/gen_stubs.py build-vulkan/logs/<link>.log vulkan/src/engine/stubs.cpp",
        "//", ""]
    body = ['#include "%s"' % include_spelling(h) for h in sorted(used)]
    body += ["#include <type_traits>", "",
             "namespace strata::vulkan_engine { [[noreturn]] void missing(const char* name); }", ""]
    out = head + body
    for qual, sh, sb, mangled, dem, info, hdr in defs:
        out.append("// %s" % qual)
        out.append("//   %s: unresolved external symbol %s" % (info["code"], mangled))
        out.append("//   reported by %s (%d mention(s))" % (info["obj"], info["mentions"]))
        out.append("//   declared by %s" % (hdr or "no header this tool can see; free function, so a definition alone names it"))
        out.append("%s { %s }" % (sh, sb))
        out.append("")
    os.makedirs(os.path.dirname(out_abs), exist_ok=True)
    open(out_abs, "w", encoding="utf-8", newline="\n").write("\n".join(out) + "\n")
    print("wrote %s : %d lines, %d stubs" % (out_path, len(out), len(defs)))


if __name__ == "__main__":
    main()
