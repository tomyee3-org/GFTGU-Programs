"""Regression checks for shell-neutral commands in the merged Random2 Help."""
import contextlib
import io
import re
import unittest

import test_merged_help as merged
import test_physics_random2 as physics_tests


class CommandContinuationTests(unittest.TestCase):
    def test_both_markers_keep_all_arguments_and_separate_commands(self):
        expected = ['python main.py --seed 12 --n_trials 2 --mean_free_path 1',
                    'python main.py --max_steps 64']
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                with self.subTest(marker=marker, newline=repr(newline)):
                    block = (f'python main.py --seed 12 {marker} \t{newline}'
                             f'  --n_trials 2 {marker}{newline}\t--mean_free_path 1{newline}'
                             'python main.py --max_steps 64')
                    commands = physics_tests.documented_commands('<pre class="narrow-break"><code>' + block + '</code></pre>')
                    self.assertEqual(commands, expected)
                    self.assertEqual([' '.join(c.split()) for c in merged.documented_command_lines(block)], expected)
                    self.assertEqual(physics_tests.command_arguments(block.split(newline + 'python')[0]),
                                     ('--seed', '12', '--n_trials', '2', '--mean_free_path', '1'))

    def test_literal_symbols_in_arguments_are_preserved(self):
        command = r'python main.py --label "x^2" --path "C:\data"'
        self.assertEqual(physics_tests.join_command_continuations(command), command)
        self.assertEqual(list(merged.documented_command_lines(command)), [command])

    def test_actual_help_commands_match_with_both_markers(self):
        source = merged.find_merged_help(merged.MODULE_DIR).read_text(encoding='utf-8')
        expected = physics_tests.documented_commands(source)
        self.assertTrue(expected)
        parser = physics_tests.cli.build_parser()
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                variant = re.sub(r'[\\^]([ \t]*\r?\n)', lambda m: marker + m.group(1), source)
                commands = physics_tests.documented_commands(variant)
                self.assertEqual(commands, expected)
                for command in commands:
                    arguments = physics_tests.command_arguments(command)
                    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                        if command in physics_tests.REJECTED_COMMANDS:
                            with self.assertRaises(SystemExit) as caught:
                                parser.parse_args(arguments)
                            self.assertEqual(caught.exception.code, 2)
                        elif '--help' in arguments or '--version' in arguments:
                            with self.assertRaises(SystemExit) as caught:
                                parser.parse_args(arguments)
                            self.assertEqual(caught.exception.code, 0)
                        else:
                            parser.parse_args(arguments)
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(variant)
                case.test_main_commands_use_recognized_options()

    def test_unknown_option_on_continued_line_is_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                command = f'python main.py --max_steps 64 {marker}\n --not_a_random2_option 3'
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page('<pre><code>' + command + '</code></pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_random2_option'):
                    case.test_main_commands_use_recognized_options()

    def test_foreign_program_filter_and_python3_are_preserved(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                page = ('<pre data-program="Other">python main.py --foreign_option 1</pre>'
                        f'<pre class="narrow-break">python3 main.py {marker}\n --max_steps 64</pre>')
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(page)
                case.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
