"""Storage backend factory with automatic Supabase->local fallback."""
from __future__ import annotations

from .local_backend import LocalCSVBackend


def make_backend(prefer: str = "auto", root: str = "results"):
    """prefer: 'auto' (Supabase if available else local), 'supabase', or 'local'."""
    if prefer in ("auto", "supabase"):
        try:
            from .supabase_backend import SupabaseBackend
            if SupabaseBackend.available():
                return SupabaseBackend()
            if prefer == "supabase":
                print("[storage] Supabase requested but unavailable; using local CSV.")
        except Exception as e:  # pragma: no cover
            print(f"[storage] Supabase init failed ({e}); using local CSV.")
    return LocalCSVBackend(root)
