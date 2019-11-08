import os
import subprocess
import time
import numpy as np
import pylab
import argparse
import glob


# if main, define args
def readArguments():
     parser=argparse.ArgumentParser("Monitor machine use over the lifetime of the command being passed")
     parser.add_argument("command",metavar="COMMAND",type=str,help="Command over whose life you want to monitor your machine",nargs="+")
     parser.add_argument("--dt",help="Time interval over which to query machine for activity. Units of s",required=False,default=60,type=float)
     parser.add_argument("--diagdir",help="Directory where diagnostsic values & plots are stored. Default is Diagnostics",required=False,default="Diagnostics")
     parser.add_argument("-v","--verbose",help="Be verbose, say everything program does. Default is False",required=False,action="store_true")
     args=parser.parse_args()
     return vars(args)

# file management defs
def MakeDiagFile(filename,diagdir="Diagnostics"):
    if os.path.isfile("%s/%s"%(diagdir,filename))==True:
        diagfile=open("%s/%s"%(diagdir,filename),"a+")
    else:
        diagfile=open("%s/%s"%(diagdir,filename),"w+")
    return diagfile
def ResetDiagFiles(diagdir="Diagnostics"):
    os.system("rm -rf %s"%diagdir)
# check pid
def check_pid(pid):
    """ Check For the existence of a unix pid. """
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    else:
        return True
# define monitoring script
t0=time.time()
def GetDiags(processid,dt=5.0):
    totalmem=float(subprocess.check_output("cat /proc/meminfo | grep MemTotal",shell=True).split(" ")[-2])
    while check_pid(processid):
        loadavg=subprocess.check_output("cat /proc/loadavg",shell=True).split(" ")[0]
        freemem=float(subprocess.check_output("cat /proc/meminfo | grep MemAvailable",shell=True).split(" ")[-2])
        memuse=100.*(1-(1.*freemem)/totalmem)
        currenttime=time.time()-t0
        memfile.write("%f %f \n"%(memuse,currenttime))
        loadfile.write("%s %f \n"%(loadavg,currenttime))
        time.sleep(dt)
# define diagnostic plots
def PlotDiags(diagfile,title,diagnostic,verbose=False,ymax=1.):
    filecontent=open(diagfile,"r").read().replace("\n","").split(" ")[0:-1]
    times=np.array(filecontent[1::2]).astype(np.float)
    # put time in appropriate units
    units="[s]"
    if np.max(times)>180:
         times=times/60.
         units="[min]"
         if np.max(times)>120:
               times=times/60.
         units="[h]"
    diag=np.array(filecontent[0::2])
    pylab.plot(times,diag)
    pylab.xlabel("Time since monitor launched %s"%units) 
    pylab.ylabel(diagnostic)
    pylab.ylim((0,ymax))
    pylab.title(title)
    pylab.tight_layout()
    pylab.savefig(diagfile[0:-3]+"png")
    pylab.clf()
    if v: print "Diagnostic plot can be found at: %s"%diagfile[0:-3]+"png"



if __name__=="__main__":
    args=readArguments()
    dt=args["dt"]
    comms=args["command"]
    diagdir=args["diagdir"]
    v=args["verbose"]
    command=""
    for i in comms:
        command+=" "+i
    if v: print "Launch command: %s"%command
    # clean up diagnostic dir
    ResetDiagFiles(diagdir)
    if os.path.isdir(diagdir)==False: os.system("mkdir %s"%diagdir)
    # start monitoring
    proc=subprocess.Popen(command,shell=True).pid
    if v: print "Process id you are monitoring: %i"%proc
    loadfile=MakeDiagFile("loadavg.txt",diagdir)
    memfile=MakeDiagFile("MemUse.txt",diagdir)
    GetDiags(dt=dt,processid=proc)
    # close diagnostic files
    loadfile.close()
    memfile.close()
    # make plots
    if v: print "Process finished. Making diagnostic plots."
    loadfilename=diagdir+"/"+"loadavg.txt"
    memfilename=diagdir+"/"+"MemUse.txt"
    ncpu=int(subprocess.check_output("echo \"$(grep -c processor /proc/cpuinfo)\"",shell=True))
    PlotDiags(loadfilename,"CPU load","NCPU",v,ncpu)
    PlotDiags(memfilename,"Memory load","Memory %",v,100)
