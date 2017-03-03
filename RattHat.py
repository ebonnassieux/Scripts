import numpy as np
import os
import sys

# get rid of the fucking idiotic futurewarning
import warnings
warnings.filterwarnings("ignore")

def ReadList(filename):
    if filename[-4:]==".npy":
        namelist=np.load(filename)
        basename=filename[-4:]
    else:
        basename=filename
        f=open(filename)
        basiclist=f.readlines()
        namelist=np.array(basiclist)
        for i in range(len(namelist)):
            namelist[i]=namelist[i].rstrip()
        np.save(basename+".npy",namelist)
    return namelist

def ChooseName(filename):
    namelist=ReadList(filename)
    # choose a name
    if len(namelist)==0:
        return None
    else:
        n=int(np.random.rand()*len(namelist))
        print "THE RATT HAT CHOOSETH: %s"%namelist[n]
        return namelist[n]

def DropName(filename):
    if filename[-4:]==".npy":
        namelist=np.load(filename)
        basename=filename[-4:]
    else:
        basename=filename

    ChosenOutname=basename+".chosen.npy"
    UnchosenOutname=basename+".unchosen.npy"
    try:
        unchosenList=np.load(UnchosenOutname)
    except IOError:
        unchosenList=np.load(basename+".npy")
    try: 
        chosenList=np.load(ChosenOutname)
    except IOError:
        chosenList=np.array([]) 

    # choose number

    try:
        name=ChooseName(UnchosenOutname)
    except IOError:
        name=ChooseName(basename)

    if name == None:
        print "RATT HAT HAS RUN OUT OF SOULS, PLEASE PREPARE NEW BATCH"
    else:
        if name in chosenList:
            print "THIS MORTAL HAS ALREADY PAID THEIR DUES"
            unchosenList=np.sort(unchosenList[unchosenList!=name])
            np.save(UnchosenOutname,unchosenList)
        else:
            unchosenList=np.sort(unchosenList[unchosenList!=name])
            chosenList=np.sort(np.append(chosenList,name))            
            np.save(UnchosenOutname, unchosenList)
            np.save(ChosenOutname,chosenList)
            print "THIS PACT IS SEALED, MORTAL"

def ASCIIwizardHat():
    print "            .             "
    print "           /:\\           "
    print "          /;:.\\          "
    print "         //;:. \\         "
    print "        ///;:.. \\        "
    print "  __--\"////;:... \\\"--__  "
    print "--__   \"--_____--\"   __-- "
    print "    \"\"\"--_______--\"\"\" "



if __name__=="__main__":
    ASCIIwizardHat()
    filename=sys.argv[1]
    DropName(filename)
