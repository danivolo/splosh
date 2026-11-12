import os, subprocess

def test_core_has_no_personal_markers():
    root = os.path.join(os.path.dirname(__file__), "..")
    r = subprocess.run(["bash", os.path.join(root, "tools", "check_split.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout
