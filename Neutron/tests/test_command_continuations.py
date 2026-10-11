"""Neutron Help checks accept Windows ^ and POSIX backslash continuations."""
import re
import shlex
import unittest

import test_merged_help as merged
from test_merged_help import Page, documented_command_lines
from test_cli_neutron import main


class CommandContinuationTests(unittest.TestCase):
    def test_both_symbols_preserve_all_arguments_and_separate_runs(self):
        expected = [['--gamma', '1.666667', '--pC', '1.26e35', '--K', '5.3802e3',
                     '--steps_per_scale', '400'], ['--output_type', 'density']]
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                for padding in ('', ' \t'):
                    with self.subTest(marker=marker, newline=repr(newline), padding=padding):
                        block = newline.join([
                            'python main.py --gamma 1.666667 ' + marker + padding,
                            '  --pC 1.26e35 --K 5.3802e3 ' + marker + padding,
                            '\t--steps_per_scale 400', '',
                            'python main.py --output_type density'])
                        commands = [shlex.split(c)[2:] for c in documented_command_lines(block)]
                        self.assertEqual(commands, expected)
                        for command in commands:
                            main.parse_args(command)

    def test_literal_symbols_inside_arguments_are_preserved(self):
        block = 'python main.py --label "A^B"\npython main.py --label \'A\\B\''
        self.assertEqual([shlex.split(c)[2:] for c in documented_command_lines(block)],
                         [['--label', 'A^B'], ['--label', 'A\\B']])

    def test_actual_help_extracts_identical_commands_for_both_symbols(self):
        path = merged.find_merged_help(merged.MODULE_DIR)
        source = path.read_text(encoding='utf-8')
        expected = [list(documented_command_lines(block)) for _, block in Page(source).pres]
        self.assertEqual(sum(map(len, expected)), 23)
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                    lambda m: marker + m.group(1), source)
                self.assertEqual([list(documented_command_lines(block))
                                  for _, block in Page(equivalent).pres], expected)
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page(equivalent)
                check.test_main_commands_use_recognized_options()

    def test_unknown_options_on_continued_lines_are_still_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page('<pre>python main.py ' + marker + '\n'
                                  '  --not_a_neutron_option 1</pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_neutron_option'):
                    check.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
