"""Binary Help checks accept Windows ^ and POSIX backslash continuations."""
import re
import shlex
import unittest

import test_merged_help as merged
from test_merged_help import Page, documented_command_lines
from test_physics_binary import HELP_HTML, documented_commands, experiment_lines


class CommandContinuationTests(unittest.TestCase):
    def assertCommands(self, block, expected):
        self.assertEqual(documented_commands('<pre>' + block + '</pre>'), expected)
        self.assertEqual([shlex.split(line)[2:] for line in
                          documented_command_lines(block)], expected)

    def test_both_symbols_join_all_arguments_and_keep_separate_runs(self):
        expected = [['--frame', 'com', '--vInitA', '60000', '--vInitB', '60000',
                     '--output_type', 'orbits'], ['--output_type', 'energy_vs_time']]
        expected_lines = ['python main.py --frame com --vInitA 60000 --vInitB 60000 --output_type orbits',
                          'python main.py --output_type energy_vs_time']
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                for padding in ('', ' \t'):
                    with self.subTest(marker=marker, newline=repr(newline), padding=padding):
                        block = newline.join([
                            'python main.py --frame com ' + marker + padding,
                            '  --vInitA 60000 --vInitB 60000 ' + marker + padding,
                            '\t--output_type orbits', '',
                            'python main.py --output_type energy_vs_time'])
                        self.assertCommands(block, expected)
                        self.assertEqual(experiment_lines('<pre>' + block + '</pre>'), expected_lines)

    def test_literal_symbols_inside_arguments_are_preserved(self):
        self.assertCommands('python main.py --frame "A^B"\n'
                            "python main.py --frame 'A\\B'",
                            [['--frame', 'A^B'], ['--frame', 'A\\B']])

    def test_shipped_help_extracts_identical_commands_for_both_symbols(self):
        expected = documented_commands(HELP_HTML)
        expected_pins = experiment_lines(HELP_HTML)
        expected_lines = [list(documented_command_lines(block))
                          for attrs, block in Page(HELP_HTML).pres
                          if attrs.get('data-program', 'Binary') == 'Binary']
        self.assertGreaterEqual(len(expected), 15)
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                    lambda match: marker + match.group(1), HELP_HTML)
                self.assertEqual(documented_commands(equivalent), expected)
                self.assertEqual(experiment_lines(equivalent), expected_pins)
                actual_lines = [list(documented_command_lines(block))
                                for attrs, block in Page(equivalent).pres
                                if attrs.get('data-program', 'Binary') == 'Binary']
                self.assertEqual(actual_lines, expected_lines)

    def test_unknown_options_on_continued_lines_are_still_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page('<pre>python main.py ' + marker + '\n'
                                  '  --not_a_binary_option 1</pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_binary_option'):
                    check.test_main_commands_use_recognized_options()

    def test_foreign_program_filter_is_preserved_for_both_symbols(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                html = ('<pre data-program="Orbit">python main.py ' + marker + '\n'
                        '  --k 1.33486e22</pre><pre>python main.py ' + marker + '\n'
                        '  --frame com</pre>')
                self.assertEqual(documented_commands(html), [['--frame', 'com']])
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page(html)
                check.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
