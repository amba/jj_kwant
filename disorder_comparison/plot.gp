#!/usr/bin/env gnuplot

gap = 200e-6 * 1.602e-19

disorder_0_data = '2021-05-11_00-17-12_JJ_width=7.5e-07_junction_length=1e-07_electrode_length=5e-06_disorder=0/free_energy.dat'

disorder_0_3_data = '2021-05-10_22-09-20_JJ_width=7.5e-07_junction_length=1e-07_electrode_length=5e-06_disorder=0.3/free_energy.dat'

disorder_1_data = '2021-05-11_10-21-18_JJ_width=7.5e-07_junction_length=1e-07_electrode_length=5e-06_disorder=1/free_energy.dat'

set grid
set xlabel 'φ / π'
set ylabel 'ε / Δ'
set key top left

stats disorder_0_data using ($1/pi):($2/gap)

plot disorder_0_data using ($1/pi):($2/gap - STATS_min_y) title 'U_{rand} = 0μ',\
     disorder_0_3_data using ($1/pi):($2/gap - STATS_min_y) title 'U_{rand} = 0.3μ' ,\
     disorder_1_data using ($1/pi):($2/gap - STATS_min_y) title 'U_{rand} = 1μ'

