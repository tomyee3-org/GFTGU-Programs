"""Log-plot regressions and focused scientific Help checks; unittest compatible."""
import contextlib
import io
import math
from pathlib import Path
import re
import sys
import unittest
from unittest import mock
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MODULE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(MODULE_DIR))
import main
import physics_atmosphere as physics
from driver_atmosphere import AtmosphereModel, AtmosphereParameters, CurveData, extract_output
from plot_atmosphere import plot_atmosphere, save_and_maybe_show
from test_merged_help import find_merged_help


def check_constant_statements(source):
    for symbol, expected in (("k_B", physics.K_BOLTZMANN), ("m_u", physics.ATOMIC_MASS_UNIT)):
        match = re.search(re.escape(symbol) + r"\s*=\s*([\d.]+)\\times10\^\{(-?\d+)\}", source)
        if match is None:
            raise AssertionError("Missing numerical Help value for " + symbol)
        actual = float(match.group(1)) * 10 ** int(match.group(2))
        if not math.isclose(actual, expected, rel_tol=1e-15):
            raise AssertionError(symbol + " does not match the physical constant")
    match = re.search(r'--mu</td>\s*<td[^>]*>\s*([\d.]+)', source)
    if match is None or float(match.group(1)) != main.parse_args([]).mu:
        raise AssertionError("Documented default mu differs from the CLI")


def check_euler_statement(source):
    start = source.index('id="beat6"')
    body = source[start:source.index('</section>', start)]
    compact = re.sub(r'\s+', '', body)
    if r'1-\Delta h/H'.replace(' ', '') not in compact:
        raise AssertionError("Missing or incorrect isothermal Euler multiplier")
    if r'1+\Delta h/H'.replace(' ', '') in compact:
        raise AssertionError("Incorrect positive isothermal Euler multiplier")


class LogScaleTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def test_cli_defaults_and_explicit_reset(self):
        self.assertFalse(main.parse_args([]).log)
        self.assertTrue(main.parse_args(['--log']).log)
        self.assertFalse(main.parse_args(['--log', '--no-log']).log)

    def test_plot_axis_changes_without_changing_data_or_altitude_scale(self):
        for mode in ('Pressure', 'Density', 'Temperature'):
            with self.subTest(mode=mode):
                args=main.parse_args([])
                model=AtmosphereModel(AtmosphereParameters(args.planet_name,args.g_accel,args.mu,args.p0,args.h_points,args.T_points,mode)).run()
                curve=extract_output(model)
                for flag in (False, True):
                    with mock.patch.object(plt, 'show'):
                        plot_atmosphere(curve, log=flag)
                    ax=plt.gcf().axes[0]
                    self.assertEqual(ax.get_xscale(),'linear')
                    self.assertEqual(ax.get_yscale(),'log' if flag else 'linear')
                    np.testing.assert_array_equal(ax.lines[0].get_xdata(),curve.x)
                    np.testing.assert_array_equal(ax.lines[0].get_ydata(),curve.y)
                    self.assertEqual(ax.get_xlabel(),curve.x_label)
                    self.assertEqual(ax.get_ylabel(),curve.y_label)
                    plt.close('all')

    def test_main_log_switch_preserves_the_entire_console_output(self):
        def run(flags):
            stream=io.StringIO()
            with mock.patch.object(plt, 'show'), contextlib.redirect_stdout(stream):
                main.main(['--h_points','0,1000','--T_points','300,300',*flags])
            self.assertEqual(plt.gcf().axes[0].get_yscale(),'log' if '--log' in flags else 'linear')
            plt.close('all')
            return stream.getvalue()
        self.assertEqual(run([]),run(['--log']))
        self.assertEqual(run([]),run(['--no-log']))

    def test_log_masks_nonpositive_values_without_modifying_the_curve(self):
        curve=CurveData([0,1,2],[10,0,-1],'Pa','altitude (m)','Pressure (Pa)','Mask test')
        with mock.patch.object(plt,'show'):
            plot_atmosphere(curve,log=True)
        ax=plt.gcf().axes[0]
        transformed=ax.yaxis.get_transform().transform(np.array(curve.y,dtype=float))
        self.assertTrue(np.isfinite(transformed[0]))
        self.assertTrue(np.all(~np.isfinite(transformed[1:])))
        self.assertEqual(curve.y,[10,0,-1])

    def test_log_without_a_positive_finite_quantity_is_a_clean_error(self):
        for values in ([],[0,-1],[math.nan,math.inf]):
            with self.subTest(values=values):
                before=plt.get_fignums()
                curve=CurveData(list(range(len(values))),values,'Pa','h','p','Invalid log')
                with self.assertRaisesRegex(ValueError,'finite positive'):
                    save_and_maybe_show(curve,show=False,log=True)
                self.assertEqual(plt.get_fignums(),before)

    def test_exercise_two_uses_the_actual_cli_switch_and_distinguishes_log_bases(self):
        source=find_merged_help(MODULE_DIR).read_text()
        body=source.split('id="exp2"',1)[1].split('</div>',1)[0]
        self.assertIn('--log',body)
        self.assertIn('--no-log',body)
        self.assertIn('32000',body)
        self.assertIn(r'\ln 10',body)
        self.assertIn('cursor',body)
        self.assertIn('Beat 7',body)

    def test_constant_statements_match_and_a_wrong_exponent_is_rejected(self):
        source=find_merged_help(MODULE_DIR).read_text()
        check_constant_statements(source)
        for original,replacement in ((r'k_B=1.380649\times10^{-23}',r'k_B=1.380649\times10^{-22}'),
                                     (r'm_u=1.66053906660\times10^{-27}',r'm_u=1.66053906660\times10^{-26}')):
            with self.subTest(original=original):
                self.assertIn(original,source)
                mutant=source.replace(original,replacement)
                with self.assertRaises(AssertionError): check_constant_statements(mutant)

    def test_euler_multiplier_matches_the_step_and_a_sign_flip_is_rejected(self):
        source=find_merged_help(MODULE_DIR).read_text()
        check_euler_statement(source)
        p=1.013e5;g=9.81;T=288.15;mu=28.97
        rho=p*mu*physics.ATOMIC_MASS_UNIT/(physics.K_BOLTZMANN*T)
        height=p/(g*rho)
        advanced=physics.hydrostatic_step(p,rho,g,height/200)
        self.assertAlmostEqual(advanced/p,0.995,places=14)
        mutant=source.replace(r'1-\Delta h/H',r'1+\Delta h/H')
        self.assertNotEqual(mutant,source)
        with self.assertRaises(AssertionError):check_euler_statement(mutant)

if __name__=='__main__':unittest.main()
