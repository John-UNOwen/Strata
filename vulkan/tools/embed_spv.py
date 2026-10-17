#!/usr/bin/env python3
"""Turn one SPIR-V module into a C header of byte arrays.

Used by vulkan/CMakeLists.txt so the probe needs no SPIR-V loader and no shader
files at run time. The engine proper will use SPIR-V the same way (small control
shaders embedded, big kernels loaded from disk) once Phase 1 exists.

usage: embed_spv.py <in.spv> <out.h> <name>
"""
import sys


def main():
    if len(sys.argv) != 4:
        print(__doc__, file=sys.stderr)
        return 2
    src, dst, name = sys.argv[1], sys.argv[2], sys.argv[3]
    with open(src, "rb") as f:
        data = f.read()
    if len(data) % 4 or len(data) < 8:
        print("embed_spv: %s is not a SPIR-V module" % src, file=sys.stderr)
        return 1
    magic = int.from_bytes(data[:4], "little")
    if magic != 0x07230203:
        print("embed_spv: %s has SPIR-V magic 0x%08x" % (src, magic), file=sys.stderr)
        return 1
    rows = []
    for i in range(0, len(data), 16):
        chunk = data[i:i + 16]
        rows.append("    " + ", ".join("0x%02x" % b for b in chunk) + ",")
    with open(dst, "w", encoding="utf-8", newline="\n") as f:
        f.write("// generated from %s by vulkan/tools/embed_spv.py, do not edit\n" % src)
        f.write("#pragma once\n\n")
        f.write("static const unsigned char %s_spv[] = {\n%s\n};\n" % (name, "\n".join(rows)))
        f.write("static const unsigned int %s_spv_len = %du;\n" % (name, len(data)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
