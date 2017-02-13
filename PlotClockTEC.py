from pylab import *
import tables
import sys


input_h5_filename = sys.argv[1]

myf=tables.openFile(input_h5_filename)

tec=myf.root.sol000.tec000.val[:]
clk=myf.root.sol000.clock000.val[:]

# if you ran clock tec separation multiple times you'll get multiple solutions for clock/tec, so take the one with the highest number (eg. tec001)

print tec.shape  # times x stations x 2 polarisations

plot(clk[:,:,0])
grid
xlabel('Time in blocks of seconds')
ylabel('Clock in seconds')
title('Clock')
show()

plot(tec[:,:,0])
grid
xlabel('Time in blocks of seconds')
ylabel('Differential in arbitrary units')
title('TEC')
show()

#even better, use masked arrays and the flags:

#flags=np.logical_not(myf.root.sol000.tec000.weigths[:])
#wtec=np.ma.array(stec,mask=np.logical_not(flags))
#plot(wtec[:,:,0])
