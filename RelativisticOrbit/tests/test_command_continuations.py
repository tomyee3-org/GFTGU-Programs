"""Regression checks for shell-neutral commands in the merged RelativisticOrbit Help."""
import contextlib
import io
import re
import unittest

import test_merged_help as merged
import test_physics_relativistic_orbit as physics_tests
import main as cli


class CommandContinuationTests(unittest.TestCase):
    def test_both_markers_keep_all_arguments_and_separate_commands(self):
        expected = ['python main.py --x_init 20000 --dt 0.001 --u_init 1e8',
                    'python main.py --max_orbits 2']
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                with self.subTest(marker=marker, newline=repr(newline)):
                    block = (f'python main.py --x_init 20000 {marker} \t{newline}'
                             f'  --dt 0.001 {marker}{newline}\t--u_init 1e8{newline}'
                             'python main.py --max_orbits 2')
                    commands = physics_tests.documented_commands('<pre class="narrow-break"><code>' + block + '</code></pre>')
                    self.assertEqual(commands, expected)
                    self.assertEqual([' '.join(c.split()) for c in merged.documented_command_lines(block)], expected)
                    self.assertEqual(physics_tests.command_arguments(block.split(newline + 'python')[0]),
                                     ('--x_init', '20000', '--dt', '0.001', '--u_init', '1e8'))

    def test_literal_symbols_in_arguments_are_preserved(self):
        command = r'python main.py --label "x^2" --path "C:\data"'
        self.assertEqual(physics_tests.join_command_continuations(command), command)
        self.assertEqual(list(merged.documented_command_lines(command)), [command])

    def test_actual_help_commands_match_with_both_markers(self):
        source = merged.find_merged_help(merged.MODULE_DIR).read_text(encoding='utf-8')
        expected = physics_tests.documented_commands(source)
        self.assertTrue(expected)
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                variant = re.sub(r'[\\^]([ \t]*\r?\n)', lambda m: marker + m.group(1), source)
                commands = physics_tests.documented_commands(variant)
                self.assertEqual(commands, expected)
                for command in commands:
                    arguments = physics_tests.command_arguments(command)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        if arguments[:2] == ('--x_init', '2000'):
                            with self.assertRaises(SystemExit) as caught:
                                cli.parse_args(arguments)
                            self.assertEqual(caught.exception.code, 2)
                        elif '--help' in arguments or '--version' in arguments:
                            with self.assertRaises(SystemExit) as caught:
                                cli.parse_args(arguments)
                            self.assertEqual(caught.exception.code, 0)
                        else:
                            cli.parse_args(arguments)
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(variant)
                case.test_main_commands_use_recognized_options()

    def test_unknown_option_on_continued_line_is_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                command = f'python main.py --max_orbits 1 {marker}\n --not_a_relativistic_option 3'
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page('<pre><code>' + command + '</code></pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_relativistic_option'):
                    case.test_main_commands_use_recognized_options()

    def test_foreign_program_filter_and_python3_are_preserved(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                page = ('<pre data-program="Other">python main.py --foreign_option 1</pre>'
                        f'<pre class="narrow-break">python3 main.py {marker}\n --max_orbits 1</pre>')
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(page)
                case.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
