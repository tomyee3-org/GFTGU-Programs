"""MercPert Help checks accept Windows ^ and POSIX backslash continuations."""
import re
import shlex
import unittest

import test_merged_help as merged
import test_physics_mercpert as legacy
from test_merged_help import Page, documented_command_lines


class CommandContinuationTests(unittest.TestCase):
    def test_both_symbols_keep_all_arguments_and_separate_commands(self):
        expected = [['--m_planet_solar', '0.1', '--vy_init', '59220',
                     '--dt', '2000', '--max_steps', '10000'], ['--max_steps', '40000']]
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                for padding in ('', ' \t'):
                    with self.subTest(marker=marker, newline=repr(newline), padding=padding):
                        block = newline.join([
                            'python main.py --m_planet_solar 0.1 ' + marker + padding,
                            '  --vy_init 59220 --dt 2000 ' + marker + padding,
                            '\t--max_steps 10000', '', 'python main.py --max_steps 40000'])
                        html = '<pre class="code-block narrow-break"><code>' + block + '</code></pre>'
                        self.assertEqual([shlex.split(c)[2:] for c in
                                          legacy.documented_commands(html)], expected)
                        self.assertEqual([shlex.split(c)[2:] for c in
                                          documented_command_lines(block)], expected)

    def test_literal_symbols_inside_arguments_are_preserved(self):
        block = 'python main.py --label "A^B"\npython main.py --label \'A\\B\''
        expected = [['--label', 'A^B'], ['--label', 'A\\B']]
        self.assertEqual([shlex.split(c)[2:] for c in
                          legacy.documented_commands('<pre>' + block + '</pre>')], expected)
        self.assertEqual([shlex.split(c)[2:] for c in documented_command_lines(block)], expected)

    def test_shipped_help_extracts_identical_commands_for_both_symbols(self):
        source = legacy.HELP_FILE.read_text(encoding='utf-8')
        expected = legacy.documented_commands(source)
        expected_lines = [list(documented_command_lines(block)) for _, block in Page(source).pres]
        self.assertEqual(len(expected), 22)
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                    lambda m: marker + m.group(1), source)
                self.assertEqual(legacy.documented_commands(equivalent), expected)
                self.assertEqual([list(documented_command_lines(block))
                                  for _, block in Page(equivalent).pres], expected_lines)
                check = legacy.BeatsHelpTests('test_every_tutorial_command_is_accepted_by_the_parser')
                check.text = equivalent
                self.assertEqual(len(check._commands(outside_beat9=True)), 18)
                self.assertEqual(len(check._commands(outside_beat9=False)), 4)

    def test_pre_commands_are_not_duplicated_by_inline_code_extraction(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                html = ('<pre class="code-block narrow-break"><code>python main.py ' + marker +
                        '\n --dt 2000</code></pre><code>python main.py --max_steps 10000</code>' +
                        '<section id="beat9"><pre>python main.py --m_planet_solar 0</pre></section>')
                check = legacy.BeatsHelpTests('test_every_tutorial_command_is_accepted_by_the_parser')
                check.text = html
                self.assertEqual(check._commands(outside_beat9=True),
                                 ['python main.py --dt 2000', 'python main.py --max_steps 10000'])
                self.assertEqual(check._commands(outside_beat9=False),
                                 ['python main.py --m_planet_solar 0'])

    def test_unknown_options_on_continued_lines_are_still_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page('<pre>python main.py ' + marker + '\n'
                                  '  --not_a_mercpert_option 1</pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_mercpert_option'):
                    check.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
