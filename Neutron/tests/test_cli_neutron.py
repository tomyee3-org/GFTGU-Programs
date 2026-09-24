"""Command-line regression checks; runnable with standard-library unittest."""
import contextlib
import io
from pathlib import Path
import math
import re
import subprocess
import sys
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import main
import physics_neutron as physics
from driver_neutron import compute_neutron_star, extract_radius_checkpoints


class CommandLineTests(unittest.TestCase):
    def test_driver_defaults_are_exposed(self):
        args = main.parse_args([])
        for name, expected in main.DEFAULTS.items():
            self.assertEqual(getattr(args, name), expected)
        self.assertEqual(args.steps_per_scale, 400)
        self.assertEqual(args.max_steps, 200_000)

    def test_selectors_and_optional_log_axis(self):
        args = main.parse_args(['--output_type', ' DENSITY ', '--log_y'])
        self.assertEqual(args.output_type, 'density')
        self.assertTrue(args.log_y)
        self.assertFalse(main.parse_args(['--no-log_y']).log_y)
        self.assertEqual(main.parse_args(['--output_type', 'mass']).output_type, 'mass')

    def test_numeric_inputs_and_limits(self):
        args = main.parse_args(['--gamma', '1.8', '--pC', '1e34', '--K', '100',
                                '--steps_per_scale', '800', '--max_steps', '10000'])
        self.assertEqual((args.gamma,args.pC,args.K,args.steps_per_scale,args.max_steps),
                         (1.8,1e34,100.0,800,10000))
        for invalid in (['--gamma', '1'], ['--K', 'nan'],
                        ['--steps_per_scale', '49'], ['--max_steps', '99'],
                        ['--output_type', 'temperature']):
            with self.subTest(invalid=invalid), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main.parse_args(invalid)

    def test_arguments_reach_solver_and_plotter(self):
        with mock.patch.object(main, 'compute_neutron_star', return_value={
            'model_version': physics.MODEL_VERSION, 'build_id': physics.BUILD_ID,
        }) as solve, mock.patch.object(main,'print_model_summary'), \
             mock.patch.object(main,'plot_neutron') as plot, \
             contextlib.redirect_stdout(io.StringIO()):
            main.main(['--gamma','1.8','--pC','1e34','--K','20',
                       '--steps_per_scale','800','--max_steps','1000',
                       '--output_type','density','--log_y'])
        solve.assert_called_once_with(1.8,1e34,20.,steps_per_scale=800,max_steps=1000)
        plot.assert_called_once_with(solve.return_value,'density',log_y=True)

    def test_version_and_cli_smoke(self):
        result = subprocess.run([sys.executable,'main.py','--version'],cwd=ROOT,
                                capture_output=True,text=True)
        self.assertEqual(result.stdout.strip(),
                         f'Neutron {physics.MODEL_VERSION} (build {physics.BUILD_ID})')
        import os
        env={**os.environ,'MPLBACKEND':'Agg'}
        result = subprocess.run([sys.executable,'main.py','--output_type','mass',
                                 '--steps_per_scale','400'],cwd=ROOT,env=env,
                                capture_output=True,text=True,timeout=60)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('Neutron-star model summary',result.stdout)
        self.assertIn('total mass',result.stdout)
        self.assertIn('central pressure',result.stdout)
        self.assertEqual(
            re.findall(r'^  (0\.00|0\.25|0\.50|0\.75|0\.90)\s+', result.stdout, re.M),
            ['0.00', '0.25', '0.50', '0.75', '0.90'],
        )

    def test_docs_build_metadata(self):
        # Neutron-claude.html (the Beats tutorial) is the Help file this
        # check requires; Neutron-original.html (the pre-Beats Reference
        # Guide) is checked only if present, since it is not shipped in a
        # flattened code-review upload and removing it in a future round
        # must not break this test. Documentation candidates are searched
        # the same way test_physics_neutron.py's find_help_file() does, so
        # this test also works from a flattened upload (Neutron-claude.html
        # beside the program modules) as well as a full sibling-repo
        # checkout. If no candidate carries Neutron-claude.html at all --
        # e.g. a code-review snapshot that omits the documentation tree
        # entirely -- the synchronization check is skipped rather than
        # failed, since there is nothing to synchronize against.
        candidates = [ROOT]
        for ancestor in (ROOT, *ROOT.parents):
            candidates.append(ancestor/'GFTGU-Documentation'/'Neutron')
            if ancestor.name != 'Neutron':
                candidates.append(ancestor/'Neutron')
        docs_dir = next(
            (c for c in candidates if (c/'Neutron-claude.html').is_file()),
            None,
        )
        if docs_dir is None:
            self.skipTest(
                'Neutron-claude.html not found beside the program or in a '
                'GFTGU-Documentation/Neutron/ tree; nothing to synchronize.'
            )
        self.assertIn(
            physics.BUILD_ID,
            (docs_dir/'Neutron-claude.html').read_text(encoding='utf-8'),
        )
        for optional in (
            docs_dir/'Neutron-ReleaseNotes.html',
            docs_dir/'SampleOutputs/Neutron-SampleOutputs_Guide.html',
            docs_dir/'Neutron-original.html',
        ):
            if optional.is_file():
                self.assertIn(
                    physics.BUILD_ID, optional.read_text(encoding='utf-8')
                )

    def test_default_solver_matches_independent_reference_and_low_pressure_case(self):
        model = compute_neutron_star(1.666667,1.26e35,5.3802e3)
        # Independent fine-grid reference from the supplied physical tests.
        self.assertAlmostEqual(model['surface_radius_m'],7802.758706219183,
                               delta=2.0)
        self.assertAlmostEqual(model['total_mass_kg'],1.9140826174028036e30,
                               delta=2e-4*1.9140826174028036e30)
        rows = extract_radius_checkpoints(model)
        self.assertEqual([row[0] for row in rows], [0.0, 0.25, 0.50, 0.75, 0.90])
        self.assertEqual((rows[0][2], rows[0][3], rows[0][4]),
                         (model['pC'], model['rhoC'], 0.0))
        self.assertTrue(all(math.isclose(row[1] * 1000.0,
                                             row[0] * model['surface_radius_m'], rel_tol=1e-14)
                            for row in rows))
        self.assertTrue(all(left[2] > right[2] and left[3] > right[3]
                            and left[4] < right[4]
                            for left, right in zip(rows, rows[1:])))
        self.assertTrue(all(math.isclose(model['K'] * row[3]**model['gamma'],
                                             row[2], rel_tol=1e-12)
                            for row in rows))
        low = compute_neutron_star(1.666667,1e27,5.3802e3)
        self.assertLess(low['total_mass_solar'],.02)
        self.assertTrue(low['causality_satisfied'])


if __name__ == '__main__':
    unittest.main()
