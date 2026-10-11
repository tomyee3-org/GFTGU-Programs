"""Regression checks for shell-neutral commands in the merged Orbit Help."""
import re
import unittest

import test_merged_help as merged
import test_physics_orbit as physics_tests


class CommandContinuationTests(unittest.TestCase):
    def test_both_markers_keep_all_arguments_and_separate_commands(self):
        expected = ['python main.py --xInit 4.6e10 --yInit 0 --vyInit 58980',
                    'python main.py --maxOrbits 2']
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                with self.subTest(marker=marker, newline=repr(newline)):
                    block = (f'python main.py --xInit 4.6e10 {marker} \t{newline}'
                             f'  --yInit 0 {marker}{newline}\t--vyInit 58980{newline}'
                             'python main.py --maxOrbits 2')
                    commands = physics_tests.documented_commands('<pre class="narrow-break"><code>' + block + '</code></pre>')
                    self.assertEqual(commands, expected)
                    self.assertEqual([' '.join(c.split()) for c in merged.documented_command_lines(block)], expected)
                    self.assertEqual(physics_tests.command_arguments(block.split(newline + 'python')[0]),
                                     ('--xInit', '4.6e10', '--yInit', '0', '--vyInit', '58980'))

    def test_literal_symbols_in_arguments_are_preserved(self):
        command = r'python main.py --label "x^2" --path "C:\data"'
        self.assertEqual(physics_tests.join_command_continuations(command), command)
        self.assertEqual(list(merged.documented_command_lines(command)), [command])

    def test_actual_help_commands_match_with_both_markers(self):
        source = merged.find_merged_help(merged.MODULE_DIR).read_text(encoding='utf-8')
        expected = physics_tests.documented_commands(source)
        self.assertTrue(expected)
        parser = physics_tests.orbit_main.build_parser()
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                variant = re.sub(r'[\\^]([ \t]*\r?\n)', lambda m: marker + m.group(1), source)
                commands = physics_tests.documented_commands(variant)
                self.assertEqual(commands, expected)
                for command in commands:
                    if not physics_tests.is_template(command):
                        parser.parse_args(physics_tests.command_arguments(command))
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(variant)
                case.test_main_commands_use_recognized_options()

    def test_unknown_option_on_continued_line_is_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                command = f'python main.py --maxOrbits 1 {marker}\n --not_an_orbit_option 3'
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page('<pre><code>' + command + '</code></pre>')
                with self.assertRaisesRegex(AssertionError, '--not_an_orbit_option'):
                    case.test_main_commands_use_recognized_options()

    def test_foreign_program_filter_and_python3_are_preserved(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                page = ('<pre data-program="Other">python main.py --foreign_option 1</pre>'
                        f'<pre class="narrow-break">python3 main.py {marker}\n --maxOrbits 1</pre>')
                case = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                case.page = merged.Page(page)
                case.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
