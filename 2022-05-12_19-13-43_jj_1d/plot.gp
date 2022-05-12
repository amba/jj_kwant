set pm3d corners2color c1
set view map
# unset surface

splot "< awk 'NF!=0 { print $0 }' data.dat" using 3:1:4