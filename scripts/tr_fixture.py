"""Reference values for src/lib/transition-radiation.ts, generated with trlib.

trlib (the TR_CL project) is the validated Python implementation of transition
radiation in cathodoluminescence:

    Ritchie & Eldridge, Phys. Rev. 126, 1935 (1962)      -- free-standing film
    Garcia de Abajo, Rev. Mod. Phys. 82, 209 (2010)      -- single interface
    Brenny, PhD thesis, UvA/AMOLF (2016), ch. 3-5
    Ebel et al., Adv. Funct. Mater. 36, e76737 (2026)    -- Eqs. 1-3, S1-S2

Every case stores the permittivity it used, so the TypeScript port is checked
against the physics alone and not against whichever optical-constant table
either side happens to ship.

usage: py -3.11 scripts/tr_fixture.py <path to TR_CL> src/lib/transition-radiation.fixture.json
"""
import json
import pathlib
import sys

SIG = 12


def main() -> None:
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    trlib_root, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    sys.path.insert(0, str(trlib_root))
    import numpy as np
    import trlib
    from trlib import single_interface as si, thin_film as tf, resonances as rs

    def r(x):
        a = np.asarray(x, dtype=float)
        return [float(f'{v:.{SIG}g}') for v in a.ravel()] if a.ndim else float(f'{a:.{SIG}g}')

    def rows(x):
        return [r(row) for row in np.asarray(x, dtype=float)]

    wl = np.linspace(400.0, 900.0, 21)                 # nm
    theta_deg = np.array([5.0, 20.0, 40.0, 60.0, 80.0])
    th = np.deg2rad(theta_deg)

    def eps_of(name):
        mat = trlib.get_material(name)
        e = np.asarray(mat(wl), dtype=complex)
        return mat, e

    cases = []

    def interface_case(cid, name, keV, side):
        mat, e = eps_of(name)
        cases.append({
            'id': cid, 'kind': 'interface', 'material': name, 'energyKeV': keV, 'side': side,
            'wavelengthNm': r(wl), 'epsRe': r(e.real), 'epsIm': r(e.imag),
            'thetaDeg': r(theta_deg),
            # photons / (electron sr nm) at each theta (rows) and wavelength (columns)
            'srPerNm': rows(si.emission_probability_sr(th[:, None], wl[None, :], keV, mat,
                                                       hemisphere='upper' if side == 'backward' else 'lower')),
            # angle-integrated over the full hemisphere, photons / (electron nm)
            'perNm': r(si.emission_probability(wl, keV, mat,
                                               hemisphere='upper' if side == 'backward' else 'lower')),
            # and over a 60 deg collection cone
            'perNmCone60': r(si.emission_probability(wl, keV, mat, theta_max=np.deg2rad(60.0),
                                                     hemisphere='upper' if side == 'backward' else 'lower')),
            'perEv': r(si.emission_probability(wl, keV, mat, per='eV',
                                               hemisphere='upper' if side == 'backward' else 'lower')),
        })

    def film_case(cid, name, d_nm, keV, side):
        mat, e = eps_of(name)
        cases.append({
            'id': cid, 'kind': 'film', 'material': name, 'thicknessNm': d_nm,
            'energyKeV': keV, 'side': side,
            'wavelengthNm': r(wl), 'epsRe': r(e.real), 'epsIm': r(e.imag),
            'thetaDeg': r(theta_deg),
            'srPerNm': rows(tf.emission_probability_sr(th[:, None], wl[None, :], d_nm, keV, mat, side=side)),
            'perNm': r(tf.emission_probability(wl, d_nm, keV, mat, side=side)),
            'perNmCone60': r(tf.emission_probability(wl, d_nm, keV, mat, side=side,
                                                     theta_max=np.deg2rad(60.0))),
            'perEv': r(tf.emission_probability(wl, d_nm, keV, mat, side=side, per='eV')),
            # the B1 / B3 interface-only control calculation of Ebel Fig. S1
            'srPerNmB1B3': rows(tf.emission_probability_sr(th[:, None], wl[None, :], d_nm, keV, mat,
                                                          side=side, terms=('B1', 'B3'))),
        })

    interface_case('interface_ag_30kev_backward', 'Ag', 30.0, 'backward')
    interface_case('interface_au_30kev_backward', 'Au', 30.0, 'backward')
    interface_case('interface_al_10kev_backward', 'Al', 10.0, 'backward')
    interface_case('interface_si_30kev_backward', 'Si', 30.0, 'backward')
    interface_case('interface_ag_30kev_forward', 'Ag', 30.0, 'forward')
    interface_case('interface_au_200kev_backward', 'Au', 200.0, 'backward')

    film_case('film_ws2_83nm_30kev_backward', 'WS2', 83.0, 30.0, 'backward')
    film_case('film_ws2_83nm_30kev_forward', 'WS2', 83.0, 30.0, 'forward')
    film_case('film_ag_50nm_30kev_backward', 'Ag', 50.0, 30.0, 'backward')
    film_case('film_si_100nm_10kev_backward', 'Si', 100.0, 10.0, 'backward')
    film_case('film_al_200nm_100kev_backward', 'Al', 200.0, 100.0, 'backward')
    film_case('film_ws2_5nm_30kev_backward', 'WS2', 5.0, 30.0, 'backward')

    # Ginzburg-Frank: the perfect-conductor limit of the single interface
    perfect = {
        'energyKeV': 30.0, 'wavelengthNm': r(wl), 'thetaDeg': r(theta_deg),
        'srPerNm': rows(si.perfect_conductor_sr(th[:, None], wl[None, :], 30.0)),
    }

    # thick film -> single interface (two independent formalisms, Ritchie vs Garcia de Abajo)
    ag_mat, ag_eps = eps_of('Ag')
    limits = {
        'thickFilmSrPerNm': rows(tf.thick_film_limit_sr(th[:, None], wl[None, :], 30.0, ag_mat)),
        'interfaceSrPerNm': rows(si.emission_probability_sr(th[:, None], wl[None, :], 30.0, ag_mat)),
        'thinFilmSrPerNm': rows(tf.thin_film_limit_sr(th[:, None], wl[None, :], 2.0, 30.0, ag_mat)),
        'filmAt2nmSrPerNm': rows(tf.emission_probability_sr(th[:, None], wl[None, :], 2.0, 30.0, ag_mat)),
        'epsRe': r(ag_eps.real), 'epsIm': r(ag_eps.imag), 'wavelengthNm': r(wl),
        'thetaDeg': r(theta_deg), 'energyKeV': 30.0,
    }

    # Ebel Eq. 3: thickness at which order m is resonant, and the Cherenkov / TIR criteria
    ws2_mat, ws2_eps = eps_of('WS2')
    resonance = {
        'material': 'WS2', 'energyKeV': 30.0, 'thetaDeg': 60.0,
        'wavelengthNm': r(wl), 'epsRe': r(ws2_eps.real), 'epsIm': r(ws2_eps.imag),
        'thicknessNmM': {str(m): r(rs.resonance_thickness(wl, 30.0, ws2_mat, m, np.deg2rad(60.0)))
                         for m in (1, 2, 3)},
        'phaseAt83nm': r(rs.interference_phase(wl, 83.0, 30.0, ws2_eps, np.deg2rad(60.0))),
    }
    cherenkov = [
        {'epsRe': 12.25, 'thresholdKeV': r(rs.cherenkov_threshold_keV(12.25)),
         'angleDeg30keV': r(np.rad2deg(rs.cherenkov_angle(30.0, 12.25))),
         'tirDeg': r(np.rad2deg(rs.tir_angle(12.25))),
         'escapes30keV': bool(rs.cherenkov_escapes(30.0, 12.25))},
        {'epsRe': 2.25, 'thresholdKeV': r(rs.cherenkov_threshold_keV(2.25)),
         'angleDeg200keV': r(np.rad2deg(rs.cherenkov_angle(200.0, 2.25))),
         'tirDeg': r(np.rad2deg(rs.tir_angle(2.25))),
         'escapes200keV': bool(rs.cherenkov_escapes(200.0, 2.25))},
    ]
    kinematics = [{'energyKeV': e, 'beta': r(trlib.beta(e))} for e in (1.0, 10.0, 30.0, 100.0, 200.0, 300.0)]

    out = {
        'source': f'trlib {trlib.__version__} (TR_CL): Ritchie & Eldridge 1962, Garcia de Abajo RMP 2010, '
                  'Brenny thesis 2016, Ebel et al. Adv. Funct. Mater. 2026',
        'generator': 'scripts/tr_fixture.py',
        'units': 'wavelength nm, electron kinetic energy keV, angles deg, '
                 'probabilities photons per electron per nm (per sr where stated)',
        'cases': cases, 'perfectConductor': perfect, 'limits': limits,
        'resonance': resonance, 'cherenkov': cherenkov, 'kinematics': kinematics,
    }
    dst.write_text(json.dumps(out, separators=(',', ':')) + '\n', encoding='utf-8', newline='\n')
    print(f'{len(cases)} cases -> {dst} ({dst.stat().st_size:,} bytes)')


if __name__ == '__main__':
    main()
