"""Ground-truth sites: Apollo landing sites with sample-based expectations.

Coordinates are the lunar module positions measured from LROC images (Wagner et al.,
2017, Icarus 283). They are given to ~10 m; check them against the paper before
quoting them. Expected mineralogy is summarised qualitatively from the returned
samples. It says what feature fitting *should* and *should not* find within a few km
of the LM, which is how the tool's thresholds are validated (see ``validate.py``).

Apollo 16 is a weak test: M3 L2 reflectance was calibrated against Apollo 16 soil 62231
(ground-truth correction, Isaacson et al., 2013), so agreement there is partly built in.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Site:
    name: str
    lat: float  # degrees N
    lon: float  # degrees E (-180..180)
    setting: str
    expect: tuple[str, ...] = field(default_factory=tuple)  # materials expected to be detected
    expect_absent: tuple[str, ...] = field(default_factory=tuple)  # materials that should not be
    notes: str = ""


APOLLO = {
    "apollo11": Site("Apollo 11", 0.6742, 23.4731, "Mare Tranquillitatis, high-Ti mare basalt",
                     expect=("High-Ca pyroxene",), expect_absent=("Mg-spinel", "Olivine"),
                     notes="Mature mare soil; pyroxene bands are weak and ilmenite darkens the spectra."),
    "apollo12": Site("Apollo 12", -3.0124, -23.4216, "Oceanus Procellarum, low-Ti mare basalt",
                     expect=("High-Ca pyroxene",), expect_absent=("Mg-spinel",),
                     notes="Olivine and pigeonite basalts; olivine may appear locally around fresh craters."),
    "apollo14": Site("Apollo 14", -3.6453, -17.4714, "Fra Mauro Formation, KREEP-rich breccias",
                     expect=("Low-Ca pyroxene",), expect_absent=("Mg-spinel",),
                     notes="Noritic (low-Ca pyroxene) breccias of Imbrium ejecta."),
    "apollo15": Site("Apollo 15", 26.1322, 3.6339, "Hadley-Apennine: mare basalt next to highland front",
                     expect=("High-Ca pyroxene", "Low-Ca pyroxene"), expect_absent=("Mg-spinel",),
                     notes="Pigeonite/augite mare basalts on the plain; noritic material on the Apennine front."),
    "apollo16": Site("Apollo 16", -8.9730, 15.5002, "Descartes highlands, feldspathic breccias",
                     expect=(), expect_absent=("High-Ca pyroxene", "Mg-spinel", "Olivine"),
                     notes="Plagioclase-rich, mafic-poor. Mature soils are nearly featureless in the NIR. "
                           "NOT an independent test: the M3 L2 ground-truth correction was derived from "
                           "Apollo 16 soil 62231 at this site (Isaacson et al., 2013)."),
    "apollo17": Site("Apollo 17", 20.1908, 30.7717, "Taurus-Littrow valley",
                     expect=("High-Ca pyroxene",), expect_absent=("Mg-spinel", "Plagioclase"),
                     notes="Valley floor: high-Ti basalt (augite, ilmenite) and dark mantle glass beads "
                           "(orange/black glass, e.g. 74220). North/South Massifs: noritic breccias "
                           "(low-Ca pyroxene), a few km from the LM."),
}
