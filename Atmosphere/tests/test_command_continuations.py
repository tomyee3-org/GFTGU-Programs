"""Help command checks accept Windows ^ and POSIX backslash continuations."""
import re
import shlex
import unittest

import test_merged_help as merged
from test_merged_help import Page, documented_command_lines
from test_physics_atmosphere import HELP_HTML, documented_commands


class CommandContinuationTests(unittest.TestCase):
    def assertCommands(self, block, expected):
        self.assertEqual(documented_commands('<pre>' + block + '</pre>'), expected)
        self.assertEqual([shlex.split(line)[2:] for line in
                          documented_command_lines(block)], expected)

    def test_both_symbols_join_all_arguments_and_keep_separate_runs(self):
        expected = [['--planet_name', 'Test planet', '--h_points', '0,8000',
                     '--T_points', '288,288', '--output_type', 'pressure'],
                    ['--output_type', 'density']]
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                for padding in ('', ' \t'):
                    with self.subTest(marker=marker, newline=repr(newline), padding=padding):
                        block = newline.join([
                            'python main.py --planet_name "Test planet" ' + marker + padding,
                            '  --h_points 0,8000 --T_points 288,288 ' + marker + padding,
                            '\t--output_type pressure', '',
                            'python main.py --output_type density'])
                        self.assertCommands(block, expected)

    def test_literal_symbols_inside_arguments_are_preserved(self):
        self.assertCommands('python main.py --planet_name "A^B"\n'
                            "python main.py --planet_name 'A\\B'",
                            [['--planet_name', 'A^B'], ['--planet_name', 'A\\B']])

    def test_shipped_help_extracts_identical_commands_for_both_symbols(self):
        expected = documented_commands(HELP_HTML)
        self.assertGreaterEqual(len(expected), 15)
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                    lambda match: marker + match.group(1), HELP_HTML)
                self.assertEqual(documented_commands(equivalent), expected)
                page = Page(equivalent)
                actual = [shlex.split(line)[2:] for _, block in page.pres
                          for line in documented_command_lines(block)]
                self.assertEqual(actual, expected)

    def test_unknown_options_on_continued_lines_are_still_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page('<pre>python main.py ' + marker + '\n'
                                  '  --not_an_atmosphere_option 1</pre>')
                with self.assertRaisesRegex(AssertionError, '--not_an_atmosphere_option'):
                    check.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
