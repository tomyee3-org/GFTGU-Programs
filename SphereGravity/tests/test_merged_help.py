"""Portable checks for the current merged Help baseline.

Run alongside the existing physical regression suite. The older suite still
contains content-specific assertions, which are retained to expose differences
between the retired tutorials and the evolving merged Help.
"""
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

PROGRAM = "SphereGravity"
MODULE_DIR = Path(__file__).resolve().parent.parent


def find_merged_help(module_dir):
    module_dir = Path(module_dir)
    for parent in (module_dir, *module_dir.parents):
        for folder in (parent, parent / "GFTGU-Documentation" / PROGRAM,
                       parent / (PROGRAM + "-Documentation"), parent / (PROGRAM + "-docs"),
                       parent / PROGRAM):
            path = folder / (PROGRAM + ".html")
            if path.is_file():
                return path
    raise FileNotFoundError(PROGRAM + ".html is required beside the program or in GFTGU-Documentation/" + PROGRAM)


class Page(HTMLParser):
    def __init__(self, source):
        super().__init__()
        self.ids, self.fragments, self.pres = [], [], []
        self.active_pre = None
        self.feed(source)
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs: self.ids.append(attrs["id"])
        if tag == "a" and attrs.get("href", "").startswith("#"):
            self.fragments.append(attrs["href"][1:])
        if tag == "pre": self.active_pre = [attrs, ""]
    def handle_data(self, data):
        if self.active_pre is not None: self.active_pre[1] += data
    def handle_endtag(self, tag):
        if tag == "pre" and self.active_pre is not None:
            self.pres.append(self.active_pre)
            self.active_pre = None


class MergedHelpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = find_merged_help(MODULE_DIR)
        cls.source = cls.path.read_text(encoding="utf-8")
        cls.page = Page(cls.source)

    def test_merged_help_is_found_without_spare_parts(self):
        self.assertEqual(self.path.name, PROGRAM + ".html")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for variant in ("short", "extended", "original", "java"):
                (folder / (PROGRAM + "-" + variant + ".html")).write_text("archive")
            with self.assertRaises(FileNotFoundError): find_merged_help(folder)
            current = folder / (PROGRAM + ".html")
            current.write_text("merged")
            self.assertEqual(find_merged_help(folder), current)

    def test_title_and_badge_use_current_house_style(self):
        title = re.search(r"<title>(.*?)</title>", self.source, re.S | re.I)
        self.assertIsNotNone(title)
        self.assertEqual(unescape(title.group(1)).strip(), PROGRAM + " – Help File")
        self.assertRegex(self.source, r'<div class="badge">Gravity from the Ground Up</div>')

    def test_visible_version_and_build_match_the_live_program(self):
        run = subprocess.run([sys.executable, "main.py", "--version"], cwd=MODULE_DIR,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        version = re.search(r"(\d+\.\d+\.\d+).*?([0-9a-f]{12})", run.stdout)
        self.assertIsNotNone(version, run.stdout)
        visible = " ".join(re.sub(r"<[^>]+>", " ", unescape(self.source)).split())
        self.assertIn("Version " + version.group(1) + " Build " + version.group(2), visible)

    def test_internal_links_have_unique_existing_targets(self):
        self.assertEqual(len(self.page.ids), len(set(self.page.ids)), "Duplicate IDs")
        missing = set(self.page.fragments) - set(self.page.ids) - {""}
        self.assertFalse(missing, "Missing anchors: " + str(sorted(missing)))

    def test_main_commands_use_recognized_options(self):
        run = subprocess.run([sys.executable, "main.py", "--help"], cwd=MODULE_DIR,
                             capture_output=True, text=True, timeout=30)
        self.assertEqual(run.returncode, 0, run.stderr)
        options = set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", run.stdout))
        checked = 0
        for attrs, block in self.page.pres:
            if attrs.get("data-program", PROGRAM) != PROGRAM: continue
            joined = re.sub(r"\\\n\s*", " ", block)
            for line in joined.splitlines():
                line = line.strip()
                if not re.match(r"python(?:3)? main\.py(?:\s|$)", line): continue
                line = line.split(" #", 1)[0]
                flags = set(re.findall(r"--[A-Za-z][A-Za-z0-9_-]*", line))
                self.assertFalse(flags - options, line + ": " + str(sorted(flags-options)))
                checked += 1
        self.assertGreater(checked, 0, "No main.py command was checked")

if __name__ == "__main__": unittest.main()
