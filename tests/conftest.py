import os
import pathlib
import sys

# Make the package importable from the repo root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

if "GEMINI_API_KEY" not in os.environ and os.name == "nt":
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            val, _ = winreg.QueryValueEx(key, "GEMINI_API_KEY")
            if val:
                os.environ["GEMINI_API_KEY"] = val
    except Exception:
        pass
