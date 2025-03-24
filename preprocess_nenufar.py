     #!/usr/bin/env python
 
import os
import glob
import numpy
 

basestr1="2020"
#basestr2="2021/01/20210107_053400_20210107_073600_COMA_PILOT00"
mslist1 = numpy.array(list(sorted(glob.glob(basestr1+'/L0/*.MS'))))
#mslist2 = glob.glob(basestr2+'/L0/*.MS')



def flagNavg(mslist):
    for myms in mslist:
        # do the thing to let dppp understand the ms set
        syscall= 'DPPP ' 
        syscall+= 'msin='+myms+' '
        syscall+= 'msin.autoweight =True '
        syscall+= 'msout='+myms+'_converted.ms ' 
        syscall+= 'steps=[] '
        syscall+= 'numthreads=56 ' 
        print (syscall) 
        os.system(syscall) 
        # flag autocorr
        syscall= 'DPPP '
        syscall+= 'msin='+myms+'_converted.ms '
        syscall+= 'msin.autoweight =True '
        syscall+= 'msout='+myms+" "
        syscall+= 'steps=[aoflag] '
        syscall+= 'aoflag.autocorr=True '
        print (syscall)
        os.system(syscall)
        # aoflag
        syscall= 'aoflagger '+myms+'_converted.ms '
        print (syscall)
        os.system(syscall)
        # average
        syscall= 'DPPP '
        syscall+= 'msin='+myms+'_converted.ms '
        syscall+= 'msout='+myms.replace("/L0/","/L1/")
        syscall+= ' numthreads=40 '
        syscall+= 'steps=[averager] '
        syscall+= 'averager.timestep=8 '
        syscall+= 'averager.freqstep=8 '
        print (syscall)
        os.system(syscall)
        # clean up
        print "rm -rf %s"%(myms)
        os.system("rm -rf %s"%(myms))
        print "rm -rf %s"%(myms+"_converted.ms")
        os.system("rm -rf %s"%(myms+"_converted.ms"))


flagNavg(mslist1)
#flagNavg(mslist2)

