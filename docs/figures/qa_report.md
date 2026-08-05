# Figure QA report

All figures were manually inspected at final physical size after strict automated validation. Each row is bound to the reviewed full-profile PNG.

| Figure | reviewed PNG SHA-256 | clipping | text size | axis honesty | symbols | grayscale ambiguity | contour readability | hero conclusion | status |
|---|---|---|---|---|---|---|---|---|---|
| figure_01_geometry_mechanism | e8c591a797567fed815ccb04befbe73da3efad81aca090da6ccce44a220015f5 | pass | pass | pass | pass | pass | pass | pass | fixed |
| figure_02_isotropic_zgv | 7721a6e676fe429041224014db29b6f06f547b48f4fdb6281c3dd232a1541833 | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_03_angular_sensitivity | 62ef2a4e6a7beed4e97e83f00794164bed746e5cdb26aada25e10e4c0d9ab8aa | pass | pass | pass | pass | pass | pass | pass | fixed |
| figure_04_morse_points | d18dbb7bfbcb95224959a50fc297e82cf8e5f108e55ed0a03f669ca594eafccf | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_05_perturbation_scaling | df3bdcea1385890b19e836f3ffc3ec6cfa71e48e122e54d6d198f2b3935cff57 | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_06_decay_crossover | 9f36cb97e0d08fbe8c3735eab7c7be0ea3a6e8dad6e4b7b435aa533dbbbee879 | pass | pass | pass | pass | pass | pass | pass | fixed |
| figure_s01_polynomial_two_element | 100b9c7028ca8c91a973ae61c1eddde88d746cfa666ca9f819f7dca893b0efe0 | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_s02_quadrature_phase | 9d16d773bbb89e96ef7dfa0e2b455263569e68174f75a09b8528f0ae79069a12 | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_s03_mode_tracking | 83e5ed1cd5dd62500634d0c560156389b85b8a7514cc0806383fb6b31657d07b | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_s04_fd_convergence | adb77f6fdd6bc0005ce1450f5bdedd87a9847800c2ea661f674d7ae7615c7950 | pass | pass | pass | pass | pass | pass | pass | fixed |
| figure_s05_source_window_sensitivity | c1346545ee793c3eab922339c07a5c256eafd085888416d559ed78b811098a04 | pass | pass | pass | pass | pass | pass | pass | pass |
| figure_s06_silicon_stress_test | 47d449dded0dcf45c88e6d06f27076109d716db53494bb77e130ae5e333a7875 | pass | pass | pass | pass | pass | pass | pass | fixed |

## Fixed items

- Figure 1: moved Morse labels and kept the count local to the registered annulus.
- Figure 3: replaced nested layout with a byte-deterministic flat constrained grid.
- Figure 6: corrected inverse-rate slope sign and separated certified/all-time errors.
- Figure S4: removed colliding finite-difference minor tick labels.
- Figure S6: moved material constants away from the stationarity curve.

## Blocked items

None.
