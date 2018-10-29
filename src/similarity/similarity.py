#!/miniconda/bin/python

# Imports

import sys
import os
import math
sys.path.append(os.path.join(sys.path[0], "../utils/"))
sys.path.append(os.path.join(sys.path[0], "../../pipeline/config"))
from checkerUtils import all_coords, coordinate2mosaic, logSystem, execAndCheck, checkJobResultsForErrors, compressJobResults
import checkerconfig
import argparse
import uuid
import h5py
import subprocess

# Arguments
SUBMITJOBANDREADY = os.path.join(
    sys.path[0], '../utils/submitJobOnClusterAndReady.py')


def calc_similarities(coordinates=None, rundir=None, infolder=None, vname='sig', vnumber=None, bgfile=None, outfile=None, granularity=100, batch_ref=10000, batch_targ=1000, local=False):

    if not outfile:

        outfile = vname + ".h5"

    if infolder is not None:
        infolder = os.path.abspath(infolder)
    else:

        versionfolder = checkerconfig.RELEASESPATH + "/" + vnumber + "/indices/"
        if not os.path.exists(versionfolder):
            os.makedirs(versionfolder)

    if rundir is not None:
        os.chdir(rundir)
    else:
        os.chdir("/aloy/scratch/mduran/mosaic/similarity/")

    if not coordinates:
        coordinates = all_coords()
    else:
        coordinates = coordinates.split("-")

    # Background distances

    if not bgfile:
        if vname == "sig":
            bgfile = "bg_distances.h5"
        else:
            sys.exit(
                "Infile %s.h5 not known yet, or background distances not known." % vname)
    else:
        pass

    log = logSystem(sys.stdout)

    # Prepare jobs

    for coordinate in coordinates:

        # Variables

        c2m = coordinate2mosaic(coordinate)

        infile = c2m + "/" + vname + ".h5"
        bgfile = c2m + "/models/" + bgfile
        filename = os.path.join(rundir, str(uuid.uuid4()))

        log.info(" Generating preparation file ")
        # external
        if infolder is not None:

            versionfolder = "%s/%s/%s/%s.h5" % (infolder, coordinate[0], coordinate, vname)

            similarity_script = os.path.dirname(
                os.path.realpath(__file__)) + "/similarity_external.py"

            with h5py.File(versionfolder + "/" + outfile, "r") as hf:
                myiks = hf["inchikeys"][:]
                V = hf["V"][:]

            with h5py.File(infile, "r") as hf:
                alreadies = set(hf["inchikeys"][:])

            idxs = [i for i, ik in enumerate(myiks) if ik not in alreadies]

            infolder_infile = filename + ".h5"

            with h5py.File(infolder_infile, "w") as hf:

                hf.create_dataset("inchikeys", data=myiks[idxs])
                hf.create_dataset("V", data=V[idxs])

            with open(filename, "w") as f:
                N = len(idxs)
                from_j = 0
                while from_j < N:
                    to_j = from_j + batch_targ
                    f.write("%s---%d---%d---%s---%s---%s---%s---%s---%s---%d\n" % (coordinate, from_j,
                                                                                   to_j, versionfolder, infile, bgfile, outfile, vname, vnumber, batch_ref))
                    from_j += batch_targ

        else:
            # internal
            similarity_script = os.path.dirname(
                os.path.realpath(__file__)) + "/similarity_internal.py"

            with h5py.File(infile, "r") as hf:
                inchikeys = hf["keys"][:]

            with h5py.File("%s/%s.h5" % (versionfolder, vname), "a") as hf:
                if coordinate in hf.keys():
                    del hf[coordinate]
                hf.create_dataset(coordinate, data=inchikeys)

            with open(filename, "w") as f:
                from_j = 0
                N = len(inchikeys)
                while from_j < N:
                    to_j = from_j + batch_targ
                    f.write("%s---%d---%d---%s---%s---%s---%s---%s---%d\n" % (coordinate,
                                                                              from_j, to_j, infile, bgfile, outfile, vname, vnumber, batch_ref))
                    from_j += batch_targ

        if local:

            with open(filename, "r") as f:
                for l in f:
                    subprocess.Popen("%s %s %s" % (
                        checkerconfig.SING_IMAGE, similarity_script, l.rstrip("\n")), shell=True).wait()

        else:

            jobName = 'sim-' + coordinate

            with open(filename, "r") as f:
                S = 0
                for l in f:
                    S += 1

            t = min(max(math.ceil(float(S) / granularity), 100), S)

            logFilename = os.path.join(rundir, jobName + ".qsub")

            scriptFile = 'singularity exec ' + checkerconfig.SING_IMAGE + \
                ' python ' + similarity_script + ' \$i '

            cmdStr = checkerconfig.SETUPARRAYJOB % {'JOB_NAME': jobName, 'NUM_TASKS': t,
                                                    'TASKS_LIST': filename,
                                                    'COMMAND': scriptFile}

            execAndCheck(cmdStr, log)

            log.info(" - Launching the job %s on the cluster " % (jobName))
            cmdStr = SUBMITJOBANDREADY + " " + rundir + " " + jobName + " " + logFilename
            execAndCheck(cmdStr, log)

            log.info(" - Checking results for the job %s  " % (jobName))
            checkJobResultsForErrors(rundir, jobName, log)
            log.info(" - Compressing output for the job %s  " % (jobName))
            compressJobResults(rundir, jobName, ['tasks'], log)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--coordinates", type=str, default=None,
                        help="Coordinates in the panel (A1...), separated by '-'. If None, all 25 CC spaces are done.")
    parser.add_argument("--rundir", type=str, default=None,
                        help="Folder where the job data will be stored")
    parser.add_argument("--infolder", type=str, default=None,
                        help="Folder where the input files are stored. Should respect the coordinate hierarchy, namely, A/A1, B/B4, etc. If None, CC-vs-CC is done.")
    parser.add_argument("--vname", type=str, default="sig",
                        help="Infile type to compute similarities with. Must be homogeneous across coordinates.")
    parser.add_argument("--vnumber", type=str, default=None,
                        help="Release number to be set")
    parser.add_argument("--bgfile", type=str, default=None,
                        help="Background distances.")
    parser.add_argument("--outfile", type=str, default=None,
                        help="Name of the output file where similarities are stored.")
    parser.add_argument("--granularity", type=int,
                        default=100, help="Tasks per job.")
    parser.add_argument("--batch_ref", type=int, default=10000,
                        help="Batch size of the reference molecules.")
    parser.add_argument("--batch_targ", type=int, default=1000,
                        help="Batch size of the target molecules (i.e. molecules per job)")
    parser.add_argument("--local", default=False, action="store_true",
                        help="Run locally (by default it runs in the cluster).")

    args = parser.parse_args()

    calc_similarities(coordinates=args.coordinates, rundir=args.rundir, infolder=args.infolder,
                      vname=args.vname, vnumber=args.vnumber, bgfile=args.bgfile, outfile=args.outfile, granularity=args. granularity,
                      batch_ref=args.batch_ref, batch_targ=args.batch_targ, local=args.local)
