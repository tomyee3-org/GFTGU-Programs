"""Multiple Help and guide checks accept Windows/POSIX continuations."""
import re
import shlex
import unittest

import test_merged_help as merged
import test_physics_multiple as legacy
from test_merged_help import Page, documented_command_lines


class CommandContinuationTests(unittest.TestCase):
    def test_both_symbols_preserve_arguments_and_separate_runs(self):
        expected = [['--n_bodies', '2', '--positions_init', '5e10,0,0;-5e10,0,0',
                     '--velocities_init', '0,25760,0;0,-25760,0', '--max_steps', '3000'],
                    ['--output_type', 'trajectories']]
        for marker in ('^', '\\'):
            for newline in ('\n', '\r\n'):
                for padding in ('', ' \t'):
                    with self.subTest(marker=marker, newline=repr(newline), padding=padding):
                        block = newline.join([
                            'python main.py --n_bodies 2 ' + marker + padding,
                            '  --positions_init "5e10,0,0;-5e10,0,0" ' + marker + padding,
                            '\t--velocities_init "0,25760,0;0,-25760,0" --max_steps 3000',
                            '', 'python main.py --output_type trajectories'])
                        for opening, closing in [('<pre>', '</pre>'),
                                                 ('<pre class="code-block narrow-break"><code>',
                                                  '</code></pre>')]:
                            self.assertEqual([shlex.split(c)[2:] for c in
                                              legacy.tutorial_commands(opening + block + closing)], expected)
                        self.assertEqual([shlex.split(c)[2:] for c in
                                          documented_command_lines(block)], expected)

    def test_literal_symbols_inside_arguments_are_preserved(self):
        block = 'python main.py --label "A^B"\npython main.py --label \'A\\B\''
        expected = [['--label', 'A^B'], ['--label', 'A\\B']]
        self.assertEqual([shlex.split(c)[2:] for c in
                          legacy.tutorial_commands('<pre>' + block + '</pre>')], expected)
        self.assertEqual([shlex.split(c)[2:] for c in documented_command_lines(block)], expected)

    def test_actual_help_and_guide_extract_identically_for_both_symbols(self):
        paths = [(legacy.HELP_PATH, 34),
                 (legacy.HELP_PATH.parent / 'SampleOutputs' / 'Multiple-SampleOutputs_Guide.html', 11)]
        for path, count in paths:
            source = path.read_text(encoding='utf-8')
            expected = legacy.tutorial_commands(source)
            self.assertEqual(len(expected), count, path.name)
            expected_lines = [list(documented_command_lines(block)) for _, block in Page(source).pres]
            for marker in ('^', '\\'):
                with self.subTest(document=path.name, marker=marker):
                    equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                        lambda m: marker + m.group(1), source)
                    self.assertEqual(legacy.tutorial_commands(equivalent), expected)
                    self.assertEqual([list(documented_command_lines(block))
                                      for _, block in Page(equivalent).pres], expected_lines)

    def test_all_guide_commands_parse_with_both_symbols(self):
        guide = legacy.HELP_PATH.parent / 'SampleOutputs' / 'Multiple-SampleOutputs_Guide.html'
        source = guide.read_text(encoding='utf-8')
        for marker in ('^', '\\'):
            equivalent = re.sub(r"[\\^]([ \t]*\r?\n)",
                                lambda m: marker + m.group(1), source)
            commands = legacy.tutorial_commands(equivalent)
            self.assertEqual(len(commands), 11)
            for command in commands:
                with self.subTest(marker=marker, command=command):
                    args = legacy.entry.parse_args(shlex.split(command)[2:])
                    self.assertEqual(args.n_bodies, len(args.masses_solar))
                    self.assertEqual(len(args.positions_init), args.n_bodies)
                    self.assertEqual(len(args.velocities_init), args.n_bodies)

    def test_unknown_options_on_continued_lines_are_still_rejected(self):
        for marker in ('^', '\\'):
            with self.subTest(marker=marker):
                check = merged.MergedHelpTests('test_main_commands_use_recognized_options')
                check.page = Page('<pre>python main.py ' + marker + '\n'
                                  '  --not_a_multiple_option 1</pre>')
                with self.assertRaisesRegex(AssertionError, '--not_a_multiple_option'):
                    check.test_main_commands_use_recognized_options()


if __name__ == '__main__':
    unittest.main()
