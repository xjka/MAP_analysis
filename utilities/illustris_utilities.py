import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.animation as animation
from matplotlib import colormaps
from matplotlib.gridspec import GridSpec
from astropy import constants as const
from astropy import units as u
import illustris_python as il
from numba import jit, njit, prange, int32, uint64, float32
from numba.typed import Dict
from numba.experimental import jitclass
import numba as nb
#from scida import load
from tqdm.autonotebook import tqdm
from pathlib import Path
#import dask.array as da
import requests
import pickle 
import h5py as h5
import re
rng = np.random.default_rng()

#import mpi4py 
#mpi4py.rc.initialize = False #don't initialize MPI automatically
#mpi4py.rc.finalize   = False #don't finalize MPI automatically
#from mpi4py import MPI



@nb.jit(nopython=True,nogil=True)
def unpackpids(pids):
    for i in range(len(pids)):
        pids[i] = (pids[i]&0x7fff) | ((pids[i]&0x7fff0000)>>1) | ((pids[i]&0x7fff00000000)>>2)
    
@nb.jit(nopython=True, nogil=True)
def unpackpid(pid):
    return (pid&0x7fff) | ((pid&0x7fff0000)>>1) | ((pid&0x7fff00000000)>>2) 

@nb.jit(nopython=True,nogil=True)
def packpid(pid):
    return (pid&0x7fff) | ((pid&0x3fff8000)<<1) | ((pid&0x1fffc0000000)<<2) 
    
    
#@nb.experimental.jitclass({'N':int32, 'map_pids':nb.types.Array(uint64,1,'C'), 'subhalo_idx':nb.types.Array(int32,1,'C')})
class HaloMapRec(object):
    def __init__(self,N,pid):#,subhalo_idx):
        self.N           = N
        self.map_pids    = [pid,]
        #self.subhalo_idcs = [subhalo_idx,]
    
    #@property
    def n_map(self):
        return self.map_pids.size
   
    def append(self,pid):#,subhalo_idx):
        self.map_pids.append(pid)
        #self.subhalo_idcs.append(subhalo_idx)    

#create dictionary of halos pids to parent halo
@nb.jit(nopython=True,nogil=True)
def add_pids(pid_dict,pids,ih):
    for pid in pids:
        pid_dict[pid] = ih

@nb.jit(nopython=True,nogil=True)
def compare(maps_tup,halo_pids,GroupNsubs,SubhaloPos,SubhaloHalfmassRad,BoxSize,GroupFirstSub, SubHaloFlag, rank,displ):
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    Nmap = len(map_pids)
    map_pids_out = np.empty(Nmap, dtype=np.uint64)
    halo_ids     = np.empty(Nmap, dtype=np.int32)
    #subhalo_ids = np.empty(Nmap, dtype=np.int32)
    cnt = 0
    for im,map_pid in enumerate(map_pids):
        if map_pid in halo_pids: #consider this a match
            ih = halo_pids[map_pid]
            map_pids_out[cnt] = map_pid
            halo_ids[cnt] = ih
            #if GroupNsubs[ih]>0:
            #    subhalo_strt_idx = GroupFirstSub[ih] 
            #    subhalo_end_idx  = subhalo_strt_idx + GroupNsubs[ih]
            #    gal_pos = SubhaloPos[subhalo_strt_idx:subhalo_end_idx,:]
            #    r100e = 3*SubhaloHalfmassRad[subhalo_strt_idx:subhalo_end_idx]
            #    dx = dist_from_com(map_struct[im,1:4], gal_pos, BoxSize, r100e)
            #    min_idx = np.argmin(dx/r100e)
            #    subhalo_idx = subhalo_strt_idx + min_idx
            #    if SubHaloFlag[subhalo_idx]:
            #        subhalo_ids[cnt] = subhalo_idx
            #    else:
            #        subhalo_ids[cnt] = -1
            #else:
            #    subhalo_ids[cnt] = -1
            cnt+=1
        
    return map_pids_out[:cnt], halo_ids[:cnt]#, subhalo_ids[:cnt]

#accounts for fact that some halos wrap
@nb.njit
def dist_from_com(pos,x_com,BoxSize,r100):
    diff = np.zeros((len(x_com),3))
    for dim in range(3):
        ds = np.abs(pos[dim]-x_com[:,dim])
        dp = BoxSize - ds
        dp[dp>r100] = np.inf
        diff[:,dim] = np.minimum(ds,dp)
    return np.sqrt(np.sum(diff**2,axis=1))

@nb.jit(nopython=True,nogil=True)
def compare_pos(map_pos, map_pids, halo_pos, halo_R, R_frac,BoxSize):
    Nmap         = len(map_pos)
    map_pids_out = np.empty(Nmap, dtype=np.uint64)
    halo_ids     = np.empty(Nmap, dtype=np.int32)
    #dists       = np.empty(Nmap, dtype=float)
    cnt = 0
    for im in range(Nmap):
        i_min = np.argmin(np.sqrt(np.sum((halo_pos-map_pos[im,:])**2, axis=1))/halo_R)
        #dist = np.linalg.norm(halo_pos-map_pos[i_min,:])#dist_from_com(map_pos[im,:],halo_pos[i_min,:],BoxSize,halo_R)
        #if dist <= R_frac: #not allowed to be further than certain fraction of halo size
            #dists[cnt]       = _dists[i_min]
        halo_ids[cnt]     = i_min
        map_pids_out[cnt] = map_pids[im]
        cnt+=1
        
    return map_pids_out[:cnt], halo_ids[:cnt]

# create dictionary relating MAPs to TNG galaxies through position
def make_map_tng_galaxy_dict_pos(halo_pos, halo_R, halo_N, maps_tup, BoxSize, zf, str_identifier, R_frac, recalc=None):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    root = 0
    gm_dict = dict()
    gm_dict_str = 'Images/gm_dict_'+str_identifier+'_z={:.2f}.pkl'.format(zf)
    if os.path.isfile(gm_dict_str) and (recalc!=True):
        if rank==root:
            
            #workaround for the fact that the pickling environment had a different name for illustris_utilities
            import acj.illustris_utilities as illutil
            import sys
            sys.modules['illustris_utilities'] = illutil
            
            print('Loading from file', end=" ")
            fb =  open(gm_dict_str,'rb')
            gm_dict = pickle.loads(fb.read())
            fb.close()
    else:
        print(rank,end=": ")
        displs = None
        sendcounts = None
        Nmap = len(maps_tup[1])
        if rank==root:
            print("Generating from sim outputs", end=" ")
            
        chunksize = int(Nmap/comm_size)
        sendcounts = np.full(comm_size, chunksize)
        for i in range(1, Nmap%comm_size+1):
            sendcounts[i]+=1
        displs = np.zeros_like(sendcounts)
        displs[1:] = np.cumsum(sendcounts[:-1])
        chunksize = sendcounts[rank]
        start = displs[rank]
        # get local slice of MAPs that this rank checks
        map_pos  = (maps_tup[1][start:start+chunksize,1:4] + BoxSize.value) % BoxSize.value 
        map_pids = maps_tup[0][start:start+chunksize]
        print("Searching...", end=" ")
        map_pids_, halo_ids_ = compare_pos(map_pos, map_pids, halo_pos, halo_R, R_frac, BoxSize.value)
        print("Done.", end=" ")
        #now need to reduce these lists to one in rank-0 
        Nel = len(map_pids_)
        sendcounts = comm.allgather(Nel)
        disps = np.zeros_like(sendcounts)
        disps[1:] = np.cumsum(sendcounts[:-1])
        print("Nel={:d}".format(Nel), end=" ")
        if rank==root:
            Nel = np.sum(sendcounts)
            map_pids   = np.empty(Nel,dtype=np.uint64)
            halo_ids   = np.empty(Nel,dtype=np.int32)
            #subhalo_ids = np.empty(Nel,dtype=np.int32)
        else:
            map_pids = None
            halo_ids = None
            #subhalo_ids = None
        comm.Gatherv(map_pids_,    [map_pids, sendcounts, disps, MPI.UINT64_T],root=root) 
        comm.Gatherv(halo_ids_,    [halo_ids, sendcounts, disps, MPI.INT32_T],root=root)
        if rank==root:
            for i,map_pid in enumerate(map_pids):
                ih = halo_ids[i]
                if ih in gm_dict:
                    gm_dict[ih].append(map_pid)#,subhalo_ids[i])
                else:
                    gm_dict[ih] = HaloMapRec(halo_N[ih],map_pid)#,subhalo_ids[i])
         
            print('Writing output to file',end=" ")
            fp = open(gm_dict_str,'wb')
            pickle.dump(gm_dict, fp, protocol=2)
            fp.close()
            print('done', end=" ")

    return gm_dict

#create dictionary relating MAPs to IllustrisTNG halos
def make_map_galaxy_dict(basePath,TNGSnapNo, maps_tup, BoxSize,mp,str_identifier,recalc=None):
    from mpi4py import MPI
    #MPI.Init()
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    root = 0 #MPI.ROOT #root rank  
    gm_dict = dict()
    gm_dict_str = 'Images/gm_dict_'+str_identifier+'.pkl'
    if os.path.isfile(gm_dict_str) and (recalc==None):
        if rank==root:
            print('Loading from file')
            fb =  open(gm_dict_str,'rb')
            gm_dict = pickle.loads(fb.read())
            fb.close()
    else:
        print(rank,end=": ")
        displs = None
        sendcounts = None
        subhalos = None
        #ParticleIDs   = None
        halos = {'GroupFirstSub':[],'GroupNsubs':[]}
        Nhalo=None
        if rank==root:
            print('Generating from sim outputs',end=' ')
            halos = il.groupcat.loadHalos(basePath, TNGSnapNo, fields=['GroupLenType','GroupFirstSub','GroupNsubs','Group_R_Mean200'])
            subhalos = il.groupcat.loadSubhalos(basePath, TNGSnapNo, fields=['SubhaloPos','SubhaloHalfmassRad','SubhaloFlag'])
            #msk = subhalos['SubhaloFlag']==1
            #subhalos['SubhaloPos'] = subhalos['SubhaloPos'][msk]
            #subhalos['SubhaloHalfmassRad'] = subhalos['SubhaloHalfmassRad'][msk]
            #del subhalos['SubhaloFlag'] #no need to pass this around
            Nhalo = halos['count']
        
        #TODO Split based on halo sizes; as it stands rank 0 has by far 
        #most work because halo sizes are sorted (mostly) large to small.
        Nhalo = comm.bcast(Nhalo, root=root)
        chunksize = int(Nhalo/comm_size)
        sendcounts = np.full(comm_size, chunksize)
        for i in range(1,Nhalo%comm_size+1):
            sendcounts[i]+=1
        displs = np.zeros_like(sendcounts)
        displs[1:] = np.cumsum(sendcounts[:-1]) 
        subhalos   = comm.bcast(subhalos,root=root)
        chunksize  = sendcounts[rank]
        GroupNsubs    = np.empty(chunksize, dtype=np.int32)
        GroupFirstSub = np.empty(chunksize, dtype=np.int32)
        comm.Scatterv([halos['GroupFirstSub'],sendcounts,displs,MPI.INT32_T], GroupFirstSub,root=root)
        comm.Scatterv([halos['GroupNsubs'],   sendcounts,displs,MPI.INT32_T], GroupNsubs,root=root)
        dmtype = il.snapshot.partTypeNum('dm')
        halo_pids = Dict.empty(key_type=uint64, value_type=int32)
        cnt = 0
        for ih in range(chunksize):
            ParticleIDs = il.snapshot.loadHalo(basePath,TNGSnapNo,displs[rank]+ih,'dm',fields=['ParticleIDs',])
            add_pids(halo_pids, ParticleIDs, displs[rank]+ih)
            cnt+=1
        if rank<comm_size-1:
            assert displs[rank]+cnt==displs[rank+1]
        else:
            assert displs[rank]+cnt==displs[rank]+sendcounts[rank]
        print("h_pids",end=" ")
        map_pids_, halo_ids_ = compare(maps_tup,halo_pids,GroupNsubs,subhalos['SubhaloPos'],subhalos['SubhaloHalfmassRad'],BoxSize,GroupFirstSub,subhalos['SubhaloFlag'],rank,displs[rank])       
        Nel = len(map_pids_)
        sendcounts = comm.allgather(Nel)
        disps = np.zeros_like(sendcounts)
        disps[1:] = np.cumsum(sendcounts[:-1])
        print("Nel={:d}".format(Nel), end=" ")
        if rank==root:
            Nel = np.sum(sendcounts)
            map_pids   = np.empty(Nel,dtype=np.uint64)
            halo_ids   = np.empty(Nel,dtype=np.int32)
            #subhalo_ids = np.empty(Nel,dtype=np.int32)
        else:
            map_pids = None
            halo_ids = None
            #subhalo_ids = None
        
        comm.Gatherv(map_pids_,    [map_pids, sendcounts, disps, MPI.UINT64_T],root=root) 
        comm.Gatherv(halo_ids_,    [halo_ids, sendcounts, disps, MPI.INT32_T],root=root)
        #comm.Gatherv(subhalo_ids_, [subhalo_ids, sendcounts, disps, MPI.INT32_T],root=root)
        if rank==root:
            for i,map_pid in enumerate(map_pids):
                ih = halo_ids[i]
                if ih in gm_dict:
                    gm_dict[ih].append(map_pid)#,subhalo_ids[i])
                else:
                    gm_dict[ih] = HaloMapRec(halos['GroupLenType'][ih,dmtype],map_pid)#,subhalo_ids[i])
         
            print('Writing output to file',end=" ")
            fp = open(gm_dict_str,'wb')
            pickle.dump(gm_dict, fp, protocol=2)
            fp.close()
            print('done', end=" ")
    
    #MPI.Finalize()
    return gm_dict



def sub_mult_hist(basePath, snapno, parent_dir,zf, NP,gal_map_dict, map_dict, maps_tup, Mp, Mp_tng, cntrl, sat, cntrl_ill, sat_ill, sel, BoxSize, figsize=(14,12), mthresh=1.8e11, MinMass=2e7):
    dmtype = il.snapshot.partTypeNum('dm') 
    stype = il.snapshot.partTypeNum('star')
    halo_masses=il.groupcat.loadHalos(basePath,snapno,fields=['GroupMassType'])[sel,dmtype]*1e10
    Nhalo = len(halo_masses)
    MinMass=MinMass
    map_central_mass = np.zeros(Nhalo)
    map_satellite = np.full((2,np.sum(sat)), np.nan)
    hidxs = np.flatnonzero(cntrl)
    cnt=0
    map_M = maps_tup[1][:,0]*Mp.value
    for ih in tqdm(hidxs): #there must be a central to be a satellite
        gm = gal_map_dict[ih]
        map_masses = np.array([map_M[map_dict[p]] for p in gm.map_pids])
        idx = np.argmax(map_masses)
        map_central_mass[ih] = map_masses[idx]
        if sat[ih]:
            keep = (np.arange(len(map_masses))!=idx)&(map_masses>mthresh)
            #assert sat[ih]==np.count_nonzero(keep), "sat[ih]:{:d}, nkeep:{:d}".format(sat[ih],np.count_nonzero(keep))
            map_satellite[0,cnt:cnt+sat[ih]] = np.repeat(halo_masses[ih],sat[ih])
            map_satellite[1,cnt:cnt+sat[ih]] = map_masses[keep]
            cnt+=sat[ih]

    gfs = il.groupcat.loadHalos(basePath,snapno,fields=['GroupFirstSub'])[sel]
    gns = il.groupcat.loadHalos(basePath,snapno,fields=['GroupNsubs'])[sel] 
    smt = il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloMassType'])[:,dmtype]*1e10
    
    tng_central_mass = smt[gfs]
    tng_central_mass[~np.array(cntrl_ill,dtype=bool)] = 0 #cntrl_ill should contain the affect of also removing gfs=-1
    
    tng_satellite = np.empty((2,np.sum(sat_ill)))
    hidxs = np.flatnonzero(sat_ill)
    sflag = il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloFlag'])\
        & (il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloMassInRadType'])[:,stype]*1e10>MinMass)

    cnt=0
    for ih in tqdm(hidxs):
        if gfs[ih]>-1:
            _slice = np.arange(gfs[ih]+1, gfs[ih] + gns[ih])
            keep = sflag[_slice] 
            nkeep = np.count_nonzero(keep)
            #assert nkeep==sat_ill[ih], "sat_ill[{:d}]:{:d}, nkeep:{:d}, GroupNsubs:{:d}".format(ih,sat_ill[ih], nkeep,gfs['GroupNsubs'][ih])
            tng_satellite[0,cnt:cnt+nkeep] = np.repeat(halo_masses[ih], nkeep)   
            tng_satellite[1,cnt:cnt+nkeep] = smt[_slice][keep] 
            cnt+=nkeep
       
    fig,ax = plt.subplots(2,2,figsize=figsize,sharex=True,gridspec_kw={'hspace':0.15,'wspace':0.1})
    
    Ns = [1,2,3,4,5]
    bins = np.logspace(np.log10(np.min(halo_masses)), np.log10(np.max(halo_masses)),90)
    idxs = np.digitize(halo_masses, bins)-1
    bin_centers = (bins[:-1]+bins[1:])/2.0
    k=0
    for N in tqdm(Ns):
        p_sat_map = np.zeros(len(bins)-1)
        p_sat_tng = np.zeros(len(bins)-1)
        for i,bc in enumerate(bin_centers):
            select = i==idxs
            nhalo = np.count_nonzero(select)
            if nhalo>0:
                p_sat_map[i] = np.count_nonzero(sat[select]==N)/nhalo
                p_sat_tng[i] = np.count_nonzero(sat_ill[select]==N)/nhalo 
        ax[0,1].plot(bin_centers, p_sat_map, '-', c=colormaps['viridis'](k/len(Ns)), label=r'$N={:d}$'.format(N))
        ax[0,1].plot(bin_centers, p_sat_tng, '--', c=colormaps['viridis'](k/len(Ns)))#, label=r'$N={:d}$'.format(N))
        k+=1

    p_sat_map = np.zeros(len(bins)-1)
    p_sat_tng = np.zeros(len(bins)-1)
    for i,bc in enumerate(bin_centers):
        select = i==idxs
        nhalo = np.count_nonzero(select)
        if nhalo>0:
            p_sat_map[i] = np.count_nonzero(sat[select])/nhalo
            p_sat_tng[i] = np.count_nonzero(sat_ill[select])/nhalo 

    ax[0,1].plot(bin_centers, p_sat_map, '-', c='gray', label=r'all'.format(N))
    ax[0,1].plot(bin_centers, p_sat_tng, '--', c='gray')#, label=r'all'.format(N))
    ax[0,1].grid()
    ax[0,1].legend()
    ax[0,1].set(ylabel=r'$p(N_{sat}=N)$', title=r'$p(N_{sat}=N)\;\;\text{vs}\;\;M_{halo}$',xscale='log', yscale='linear')
    

    ax[0,0].plot(halo_masses, map_central_mass, '.', ms=3, alpha= 0.7, label='MAP')
    ax[0,0].plot(halo_masses, tng_central_mass, '.', ms=3, alpha=0.5, label='TNG')
    ax[0,0].grid()
    ax[0,0].legend(loc='upper left')
    ax[0,0].set(ylabel=r'$M_{central}\;\;[M_{\odot}]$', title=r'$M_{central}\;\;vs.\;\;M_{halo}$', xscale='log', yscale='log')

    ax[1,0].plot(map_satellite[0,:], map_satellite[1,:], '.', ms=3, alpha=0.7, label='MAP')
    ax[1,0].plot(tng_satellite[0,:], tng_satellite[1,:], '.', ms=3, alpha=0.5, label='TNG')
    ax[1,0].grid()
    ax[1,0].legend(loc='upper left')
    ax[1,0].set(xlabel=r'$M_{halo}\;\; [M_{\odot}]$', ylabel=r'$M_{sat}\;\;[M_{\odot}]$', title=r'$M_{sat}\;\;vs.\;\;M_{halo}$', xscale='log', yscale='log')
                     
    plt.show()


def halo_mass_profiles_sample(basePath,snapno,gm_dict,maps_tup,map_dict): 
    # get vector of subhalo masses 
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    ptype = il.snapshot.partTypeNum('stars')
    halosizes = il.groupcat.loadHalos(basePath, snapno, fields=['GroupLenType'])[:,ptype]  
    bins = np.logspace(np.log10(np.min(halosizes)), np.log10(np.max(halosizes)), 21)
    binidcs = np.digitize(halosizes, bins)-1
    MAP_masses = []
    for binid in [2,7,12,18]:
        ids = (binidcs==binid).nonzero()
        for ih in ids:
            if ih in gm_dict:
                for i,mpid in enumerate(gm_dict[ih].map_pids):
                    mpid = gm_dict[ih].map_pids[i]
                    im = map_dict[mpid]
                    MAP_masses.append(map_struct[im,0])

    print(len(MAP_masses))
    fig,ax = plt.subplots(1,figsize=(11,6))
    ax.scatter(subhalo_masses, MAP_masses, s=1)
    ax.grid()
    ax.set(xscale='log', yscale='log', xlabel=r'TNG300 Sub-halos N', ylabel=r'MAPs N')




def radial_profiles(cat, hm_dict, basePath,snapno,gal_map_dict, maps_tup,map_dict, Mp, nr=3,nc=3,figsize=(9,6), map_thresh=3.7e11, tng_thresh=1.5e9):
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    ptype    = il.snapshot.partTypeNum('stars')
    dmtype   = il.snapshot.partTypeNum('dm')
    halos    = il.groupcat.loadHalos(basePath, snapno, fields=['GroupMassType','GroupNsubs','GroupFirstSub', 'GroupPos'])
    subhalos = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloMassType','SubhaloFlag', 'SubhaloPos'])
    sortidcs = np.flip(np.argsort(halos['GroupMassType'][:,dmtype]))

    nbins = 15
    rhist_edges = np.logspace(np.log10(6e-2), np.log10(10), nbins+1)
    rhist = rhist_edges[:-1] + np.diff(rhist_edges)/2.
    
    rhonfw = np.zeros(nbins)
    rhomap = np.zeros(nbins)

    fig,ax = plt.subplots(nr,nc,figsize=figsize,gridspec_kw={'hspace':0.0,'wspace':0.0}, sharex=True, sharey=False)
    for ii in tqdm(range(nr*nc)):
        row = ii//nc
        col = ii%nc 
        idx = sortidcs[100*ii]
        while idx not in hm_dict:
            idx+=1
        MAP_masses    = np.empty(len(hm_dict[idx].map_pids))
        MAP_radii     = np.empty(len(hm_dict[idx].map_pids))
        for i,mpid in enumerate(hm_dict[idx].map_pids):
            im = map_dict[mpid]
            MAP_radii[i]  = np.linalg.norm(cat.halos['x_com'][idx] - map_struct[im,1:4])
            MAP_masses[i] = map_struct[im,0]*Mp.value
            
        MAP_radii = np.histogram(MAP_radii, bins=rhist_edges, density=True,weights=MAP_masses)[0]
        tng_radii = np.linalg.norm(subhalos['SubhaloPos'][halos['GroupFirstSub'][idx]:halos['GroupFirstSub'][idx]+halos['GroupNsubs'][idx],:] - halos['GroupPos'][idx], axis=1)/1e3
        subsizes = subhalos['SubhaloMassType'][halos['GroupFirstSub'][idx]:halos['GroupFirstSub'][idx]+halos['GroupNsubs'][idx],ptype]
        msk = subhalos['SubhaloFlag'][halos['GroupFirstSub'][idx]:halos['GroupFirstSub'][idx]+halos['GroupNsubs'][idx]] & (subsizes>tng_thresh/1e10)
        tng_radii = np.histogram(tng_radii[msk], bins=rhist_edges, density=True, weights=subsizes[msk])[0]

        ax[row,col].plot(rhist, tng_radii, 'r.-', label='TNG')
        ax[row,col].plot(rhist, MAP_radii, 'b.-', label='MAP')
        ax[row,col].grid()
        #ax[row,col].legend()
        ax[row,col].set(xscale='linear', yscale='log')
        man,exp = get_sci_not(halos['GroupMassType'][idx,dmtype]*1e10)
        ax[row,col].text(0.2, 0.5, r"$M_{{Halo}}={:.1f}\times 10^{{{:d}}}\;\;[h^{{-1}} M_{{\odot}}]$ ".format(man, exp)) #r"$\log_{10}(\frac{M}{M_{\odot}})=$")
    
    fig.supxlabel('N')
    fig.supylabel('counts')
    handles,labels = [], []
    for h,l in zip(*ax[nr-1,nc-1].get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels)

def halo_mass_times(basePath,snapno,gm_dict,maps_tup,map_dict,merger_tree,nr=3,nc=3,figsize=(9,6)):
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    ptype = il.snapshot.partTypeNum('stars')
    dmtype = il.snapshot.partTypeNum('dm')
    halos = il.groupcat.loadHalos(basePath, snapno, fields=['GroupLenType','GroupNsubs','GroupFirstSub'])
    sortidcs = np.flip(np.argsort(halos['GroupLenType'][:,dmtype]))
    fig,ax= plt.subplots(nr,nc,figsize=figsize,gridspec_kw={'hspace':0.0,'wspace':0.0},sharex=True, sharey=True)
    for ii in tqdm(range(nr*nc)):
        row = ii//nc
        col = ii%nc 
        idx = sortidcs[100*ii]
        while idx not in gm_dict:
            idx+=1
        MAP_masses           = np.empty(len(gm_dict[idx].map_pids))
        MAP_formation_times  = np.empty(len(gm_dict[idx].map_pids))
        MAP_halfmass_times   = np.empty(len(gm_dict[idx].map_pids))
        MAP_merger_frac      = np.zeros(len(gm_dict[idx].map_pids))
        for i,mpid in enumerate(gm_dict[idx].map_pids):
            im = map_dict[mpid]
            MAP_masses[i] = map_struct[im,0]
            MAP_formation_times[i] = merger_tree.mainbranch_formation_time(mpid)
            MAP_halfmass_times[i]  = merger_tree.mainbranch_halfmass_time(mpid)
            zm,masses,_,_          = merger_tree.largest_mainbranch_merger(mpid)
            if masses!=None:
                MAP_merger_frac[i]     = masses[1]/masses[0]

        ax[row,col].scatter(MAP_halfmass_times, MAP_masses, marker='*', s=4, c=colormaps['plasma'](MAP_merger_frac), label='half-mass time')
        ax[row,col].scatter(MAP_formation_times, MAP_masses, s=4, c=MAP_merger_frac,  label='formation time')
        ax[row,col].grid()
        #ax[row,col].legend()
        ax[row,col].set(xscale='log', yscale='log')
        ax[row,col].text(2e1,1.5e2,r"$N_{{Halo}}={:d}$".format(halos['GroupLenType'][idx,dmtype]))#r"$\log_{10}(\frac{M}{M_{\odot}})=$")
    ax[row,col].invert_xaxis() #need to do outside loop so that doesn't depend on number of loop iterations being odd (since sharex=True)
    fig.supxlabel(' time [z]')
    fig.supylabel('N')
    handles,labels = [], []
    for h,l in zip(*ax[nr-1,nc-1].get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    fig.legend(handles, labels)

def get_particle_pos_tng(basePath,snapno,partType,max_p=5e7, return_vel=False):
    pos = np.empty((int(max_p),3),dtype=float)
    vel = np.empty((int(max_p),3),dtype=float)
    subset = il.snapshot.getSnapOffsets(basePath, snapno, 0, "Subhalo")
    with h5.File(il.snapshot.snapPath(basePath, snapno), 'r') as f:
        header = dict(f['Header'].attrs.items())
        nPart = il.snapshot.getNumPart(header)
    downsample = max_p/nPart[il.snapshot.partTypeNum(partType)]
    startpoint=0
    chunk_len=500000
    totstored=0
    while(startpoint < nPart[il.snapshot.partTypeNum(partType)]):
        len_range=np.min([chunk_len, nPart[il.snapshot.partTypeNum(partType)]-startpoint])
        subset['offsetType'][il.snapshot.partTypeNum(partType)] = startpoint
        subset['lenType'][il.snapshot.partTypeNum(partType)]    = len_range
        pos_tmp = il.snapshot.loadSubset(basePath, snapno, partType, fields=['Coordinates'], subset=subset)
        startpoint=startpoint+len_range
        n2store = int(np.floor(downsample*len_range))
        pos[totstored:totstored+n2store,:] = pos_tmp[rng.choice(len_range, size=n2store)]
        if return_vel:
            vel[totstored:totstored+n2store,:] = pos_tmp[rng.choice(len_range, size=n2store)]
        totstored += n2store
    if return_vel:
        return pos[:totstored,:], vel[:totstored,:]
    else:
        return pos[:totstored,:]

