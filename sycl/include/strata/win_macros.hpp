// include/strata/win_macros.hpp - Windows port: windows.h (minwindef.h/rpcndr.h) defines small, mid, large,
// near, far, OUT as macros. Engine translation units use those words as identifiers. Include after every
// header so the names are usable for the rest of the file; the SDK chain is already parsed and include-
// guarded, so the undefs cannot break it.
#pragma once
#if defined(_WIN32)
#ifdef near
#undef near
#endif
#ifdef far
#undef far
#endif
#ifdef huge
#undef huge
#endif
#ifdef small
#undef small
#endif
#ifdef mid
#undef mid
#endif
#ifdef large
#undef large
#endif
#ifdef OUT
#undef OUT
#endif
#endif
