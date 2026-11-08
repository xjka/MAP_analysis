def mass_vs_z(parent_dir,str_identifier,BoxSize, z_end=0.0, recalc=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    root = 0

    #load data however you have to
    gal_dict_str   = 'Images/gal_dicts_'+str_identifier+'.pkl'
    if os.path.isfile(gal_dict_str) and not recalc:
        if rank==root:
            print('Loading from file')
            with open(gal_dict_str,'rb') as fb:
                gal_dicts = pickle.loads(fb.read())
        else:
            gal_dicts = None
    else:
        subdirs = None
        if rank==root:
            print('Generating from sim outputs', end=" ")
            z,subdirs = get_groupentry_subdirs(parent_dir,z_end)

        #get this rank's sub-directories
        subdirs = comm.bcast(subdirs, root=root)
        Ns = len(subdirs)
        rank_n_dirs = np.full(comm_size, Ns//comm_size)
        if Ns%comm_size!=0:
            rank_n_dirs[-Ns%comm_size:] += 1
        rank_first_dir = np.hstack(([0,], np.cumsum(rank_n_dirs[:-1])))
        ns = rank_n_dirs[rank]
        fs = rank_first_dir[rank]
        my_subdirs = subdirs[fs:fs+ns]

        # Do processing
        if rank==root:
            Nfiles=len(subdirs) #for root, will write at the beggining of array, so can allocate space for all results now and write to them from others ranks at the end
        else:
            Nfiles = len(my_subdirs)
        
        nrbins = 30
        r_bin_edges = np.logspace(np.log10(2e-2), np.log10(30), nrbins+1) 
        r_bins = r_bin_edges[:1]+np.diff(r_bin_edges)/2.
        gal_new =     {'gal_mass':np.zeros(Nfiles,dtype=np.float32), 'n_gal':np.zeros(Nfiles,dtype=np.float32),     'v_rms':np.zeros(Nfiles,dtype=np.float32)}
        gal_accrete = {'gal_mass':np.zeros(Nfiles,dtype=np.float32), 'n_gal':np.zeros(Nfiles,dtype=np.float32),     'v_rel':np.zeros(Nfiles,dtype=np.float32), \
                       'accr_mult':np.zeros(Nfiles,dtype=np.float32)}
        gal_merge =   {'gal_mass':np.zeros(Nfiles,dtype=np.float32), 'n_gal':np.zeros(Nfiles,dtype=np.float32),     'merged_gal':np.zeros(Nfiles,dtype=np.float32), \
                       'v_rel':np.zeros(Nfiles,dtype=np.float32)}
        gal_combo =   {'gal_mass':np.zeros(Nfiles,dtype=np.float32), 'n_gal_accr':np.zeros(Nfiles,dtype=np.float32),'n_gal_merge':np.zeros(Nfiles,dtype=np.float32)}
        
        for i,subdir in enumerate(my_subdirs):
            filelist = get_groupentry_filelist(subdir)
            groups = {}
            for file in filelist:
                #go through this file
                GE = np.fromfile(file, dtype=GroupEntry_dtype)
                #process the group entries
                if(len(GE)>0):
                    for ge in GE:
                        ge_m = int(ge['m'])
                        group_no = ge['group_no']
                        if group_no in groups:
                            groups[group_no]['m'].append(ge_m)
                            groups[group_no]['n_sng'] += int(ge_m==1)
                            groups[group_no]['n_gal'] += int(ge_m>1)
                            groups[group_no]['vel'].append(bitpacked.unpack_rvint(ge['rv'], BoxSize, float_dtype=np.float32)[1])
                        else:
                            vel = bitpacked.unpack_rvint(ge['rv'], BoxSize, float_dtype=np.float32)[1]
                            groups[group_no] = {'m':[ge_m,],'n_sng':int(ge_m==1), 'n_gal':int(ge_m>1), 'vel':[vel,]}

            for group_no, group in groups.items():
                group['m']   = np.array(group['m'])
                v = np.array(group['vel']).reshape(-1,3)
                if group['n_gal']==0: #new formed galaxy 
                    m = np.sum(group['m'])
                    gal_new['gal_mass'][i] += m 
                    gal_new['n_gal'][i] += 1
                    vcom = np.average(v, weights=group['m'], axis=0).reshape(-1,3)
                    v = v - vcom 
                    gal_new['v_rms'][i] += np.sum(v**2)
                elif group['n_sng']==0: #galaxy merger
                    gal_merge['n_gal'][i] += 1
                    gal_merge['merged_gal'][i] += group['n_gal']
                    gal_merge['gal_mass'][i] += np.sum(group['m'])
                    Lv = v.shape[0]
                    dv = 0
                    for _i in range(1,Lv):
                        for _j in range(_i):
                            dv += np.linalg.norm(v[_i,:]-v[_j,:])  
                    gal_merge['v_rel'][i] += dv / (Lv*(Lv-1)/2.) 
                elif group['n_gal']==1 and group['n_sng']>=1: #pure galaxy accretion
                    gal_accrete['n_gal'][i] += 1
                    gal_accrete['gal_mass'][i] += np.sum(group['m'][group['m']==1]) #gal_mass is misnomer, is accreted mass
                    gal_accrete['v_rel'][i] += np.sum((v[group['m']==1,:] - v[group['m']>1,:])**2)
                    gal_accrete['accr_mult'][i] += group['n_sng']
                else: #combination of merging and accretion
                    gal_combo['n_gal_accr'][i] += np.sum(group['m'][group['m']==1])
                    gal_combo['gal_mass'][i] += np.sum(group['m'][group['m']>1])
                    gal_combo['n_gal_merge'][i] += len(group['m'][group['m']>1])
                
                
            if gal_new['gal_mass'][i]>0:
                gal_new['v_rms'][i] = (gal_new['v_rms'][i]/gal_new['gal_mass'][i])**0.5
            if gal_merge['n_gal'][i]>0:
                gal_merge['v_rel'][i] /= gal_merge['n_gal'][i]
            if gal_accrete['accr_mult'][i]>0:
                gal_accrete['v_rel'][i]  =  (gal_accrete['v_rel'][i] / gal_accrete['accr_mult'][i])**0.5
                gal_accrete['accr_mult'][i] /= gal_accrete['n_gal'][i]

        #now need to reduce the results from all ranks
        if rank==root:
            comm.Gatherv(MPI.IN_PLACE, [gal_new['gal_mass'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_new['n_gal'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_new['v_rms'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_accrete['gal_mass'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_accrete['n_gal'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_accrete['v_rel'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_accrete['accr_mult'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_merge['gal_mass'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_merge['n_gal'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_merge['merged_gal'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_merge['v_rel'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_combo['gal_mass'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_combo['n_gal_accr'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(MPI.IN_PLACE, [gal_combo['n_gal_merge'], rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
        else:
            comm.Gatherv(gal_new['gal_mass'],     [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root) 
            comm.Gatherv(gal_new['n_gal'],        [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_new['v_rms'],        [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_accrete['gal_mass'], [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_accrete['n_gal'],    [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_accrete['v_rel'],    [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_accrete['accr_mult'],[None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_merge['gal_mass'],   [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_merge['n_gal'],      [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_merge['merged_gal'], [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_merge['v_rel'],      [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_combo['gal_mass'],   [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_combo['n_gal_accr'], [None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
            comm.Gatherv(gal_combo['n_gal_merge'],[None, rank_n_dirs, rank_first_dir, MPI.FLOAT], root=root)
        
        if rank==root:
            print('Saving to files', end=" ")
            gal_dicts = {'new':gal_new, 'merge':gal_merge, 'accrete':gal_accrete, 'combo':gal_combo, 'z':z}
            with open(gal_dict_str, 'wb') as fp:
                pickle.dump(gal_dicts,fp)
            print('done', end=" ")
        else:
            gal_dicts = None
            
    return gal_dicts


def plot_formation_hist(gal_dicts,NP,str_identifier,figsize=(13,7.5)):
    gal_new = gal_dicts['new']
    gal_merge = gal_dicts['merge']
    gal_accrete = gal_dicts['accrete']
    gal_combo  = gal_dicts['combo']
    z = gal_dicts['z']
    fig,axes = plt.subplots(1,2,figsize=figsize,gridspec_kw={'hspace':0.12}, sharex=True)
    ms=8
    lw=3.5
    axes[0].set(title=r'$total\;\; N_{MAP}$ per time step', ylabel=r'$N_{MAP}$', yscale='log')
    axes[0].plot(z, gal_new['gal_mass'],'-',ms=ms, lw=lw, label='formation')
    axes[0].plot(z, gal_accrete['gal_mass'],'-',ms=ms, lw=lw, label='accreted')
    #axes[0].plot(z, gal_merge['gal_mass'],'-',ms=ms, lw=lw, label='merge')
    axes[0].plot(z, np.array(gal_new['gal_mass'])+np.array(gal_accrete['gal_mass']),'-', lw=lw, ms=ms, color='dimgray', label='total')
    axes[0].legend()
    axes[0].grid()

    formation = np.cumsum(np.flip(gal_new['gal_mass']))/NP*100
    accretion = np.cumsum(np.flip(gal_accrete['gal_mass']))/NP*100
    axes[1].set(title=r'Fraction of mass in MAPs over time', xscale='log', xlabel='z')
    axes[1].plot(np.flip(z), formation, '-', lw=4, label='formation')
    axes[1].plot(np.flip(z), accretion, '-', lw=4, label='accreted')
    #axes[1].plot(np.fliptotal (z), np.cumsum(np.flip(gal_combo['n_gal_accr']))/NP*100, '--', label='accreted (w/ merge)')
    #axes[2].plot(z, np.cumsum(gal_accrete['accreted_mass'])/NP*100+
    #             np.cumsum(gal_combo['accreted_mass'])/NP*100, label='accreted tot')
    axes[1].plot(np.flip(z), formation+accretion+np.cumsum(np.flip(gal_combo['n_gal_accr']))/NP*100, '.-', lw=4, color='dimgray', label='total')
    axes[1].legend(loc='lower right')
    axes[1].invert_xaxis()
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[1].grid()
    #axes[1].text(10,35,r'Fraction of mass in MAPs over time')
    return fig

def plot_formation_events(gal_dicts,NP,str_identifier,figsize=(7.5,6.5)):
    gal_new = gal_dicts['new']
    gal_merge = gal_dicts['merge']
    gal_accrete = gal_dicts['accrete']
    gal_combo  = gal_dicts['combo']
    z = gal_dicts['z']
    lw=3.5
    fig,ax= plt.subplots(1,figsize=figsize)
    ax.set(title=r'Number of events per time step', xlabel='z', ylabel=r'$N_{event}$', xscale='log', yscale='log')
    ax.plot(z, gal_new['n_gal'],'-', lw=lw, label='formations')
    ax.plot(z, gal_accrete['n_gal'],'-', lw=lw, label='accretions')
    ax.plot(z, gal_merge['n_gal'],'-', lw=lw, label='mergers')
    #ax.plot(z, gal_accrete['accr_mult'], '-', lw=lw, label=r'$\langle N_{accr}\rangle$')
    #ax.plot(z, np.array(gal_merge['merged_gal'])/np.array(gal_merge['n_gal']), lw=lw, label=r'merge multiplicity')    
    ax.invert_xaxis()
    ax.legend(bbox_to_anchor=(0.65,0.5), loc='upper left')
    ax.grid()
    return fig

def plot_formation_vel(gal_dicts,BoxSize, Omega_M, H0, Mp,NP,str_identifier,figsize=(7.5,6.5)):
    gal_new = gal_dicts['new']
    gal_merge = gal_dicts['merge']
    gal_accrete = gal_dicts['accrete']
    gal_combo  = gal_dicts['combo']
    z = gal_dicts['z']
    fig,ax = plt.subplots(1, figsize=figsize)
    mu=0.59
    lw=3
    nc = 0.65290
    rho_crit = (3*H0**2/(8*np.pi*G)).to(u.Msun/u.Mpc**3).value
    ln_rho = np.log(nc/(0.55*0.12)**3*Omega_M * rho_crit * (1+z)**3)
    C3 = 1.87044114e-03 
    C2 = -1.68114168e-01 
    C1 =  5.14122638e+00 
    C0 =  -4.80177469e+01 
    vlim = np.exp(C3*ln_rho**3 + C2*ln_rho**2 + C1*ln_rho + C0)

    ax.set_title('Velocity of events at each time step')
    ax.plot(z, gal_new['v_rms'],     lw=lw, label=r'new $\langle v_{rms} \rangle$')
    ax.plot(z[z<3], vlim[z<3],  '--', lw=lw, c='tab:gray') 
    ax.plot(z, gal_accrete['v_rel'], lw=lw, label=r'accreted $\langle v_{rel} \rangle$')
    ax.plot(z, gal_merge['v_rel'],   lw=lw, label=r'merge $\langle v_{rel} \rangle$')
    ax.invert_xaxis()
    ax.set(xscale='log',yscale='linear', ylabel='km/s', xlabel='z')
    ax.legend()
    ax.grid()
    ax1 = ax.secondary_yaxis('right', functions=(lambda v: (0.5*mu*m_p*(v*u.km/u.s)**2/k_B).to(u.K).value, lambda T: np.sqrt((2*(T*u.K)*k_B/mu/m_p).to(u.km**2/u.s**2).value)))
    ax1.set_ylabel(r'T [K]',rotation=270)
    ax1.set(yscale='log')
    return fig
    
def make_mass_vs_z_plot(gal_dicts,NP,str_identifier,figsize=(12,10.5)):
    gal_new = gal_dicts['new']
    gal_merge = gal_dicts['merge']
    gal_accrete = gal_dicts['accrete']
    gal_combo  = gal_dicts['combo']
    z = gal_dicts['z']
    fig = plt.figure(figsize=figsize)
    gs  = GridSpec(3, 2, width_ratios=[1,1], wspace=0.05,hspace=0.2)
    axes = []
    axes.append(fig.add_subplot(gs[0, 0]))
    axes.append(fig.add_subplot(gs[1, 0]))
    axes.append(fig.add_subplot(gs[2, :]))
    axes.append(fig.add_subplot(gs[:2, 1]))

    axes[0].set(title=r'number of galaxies per time step', ylabel=r'$N_{galaxies}$', xscale='log', yscale='log')
    axes[0].plot(z, gal_new['n_gal'],'.', label='new galaxies')
    axes[0].plot(z, gal_accrete['n_gal'],'.', label='accreted galaxies')
    axes[0].plot(z, gal_accrete['accr_mult'], '.', label=r'$\langle N_{accr}\rangle$')
    axes[0].plot(z, gal_merge['n_gal'],'-', label='merge events')
    axes[0].plot(z, np.array(gal_merge['merged_gal'])/np.array(gal_merge['n_gal']), label=r'merge multiplicity')
    #axes[0].plot(z, gal_combo['n_gal'],'-',label='merge/accr. galaxies')
    axes[0].invert_xaxis()
    axes[0].legend()
    axes[0].grid()


    axes[1].set(title=r'mass per time step', ylabel=r'$M_{galaxies}$', xscale='log', yscale='log')
    axes[1].plot(z, gal_new['gal_mass'],'.',label='new galaxies')
    axes[1].plot(z, gal_accrete['gal_mass'],'.',label='accreted mass')
    #axes[1].plot(z, gal_merge['gal_mass'],'-',label='merged galaxies')
    #axes[1].plot(z, gal_combo['gal_mass'],'-', label='merge/accr. galaxies')
    #axes[1].plot(z, gal_combo['accreted_mass'],'.-',label='merge/accr. accreted mass')
    axes[1].invert_xaxis()
    axes[1].legend()
    axes[1].grid()

    axes[2].set(title=r'fraction of mass in galaxies over time',xscale='log', xlabel='z')
    #TODO: formation and accetion cumsums
    #axes[2].plot(z, np.array([np.sum(m) for m in M])/NP*100,'.-',color='tab:gray', label='total')
    formation = np.cumsum(np.flip(gal_new['gal_mass']))/NP*100
    accretion = np.cumsum(np.flip(gal_accrete['gal_mass']))/NP*100
    axes[2].plot(np.flip(z), formation, '-', label='formation')
    axes[2].plot(np.flip(z), accretion, '-', label='accreted')
    axes[2].plot(np.flip(z), np.cumsum(np.flip(gal_combo['n_gal_accr']))/NP*100, '--', label='accreted (w/ merge)')
    #axes[2].plot(z, np.cumsum(gal_accrete['accreted_mass'])/NP*100+
    #             np.cumsum(gal_combo['accreted_mass'])/NP*100, label='accreted tot')
    axes[2].plot(np.flip(z), formation+accretion+np.cumsum(np.flip(gal_combo['n_gal_accr']))/NP*100, '.-', color='tab:gray', label='total')
    axes[2].legend()
    axes[2].invert_xaxis()
    axes[2].yaxis.set_major_formatter(mtick.PercentFormatter())
    axes[2].grid()
    plt.show()

    fig.savefig('Images/Mass_vs_z_'+str_identifier+'.png', bbox_inches='tight')




MergerTreeNode = np.dtype([('pid',np.uint64),('N',np.int32),('zf',np.float32),('sibling',np.int32),('main_progenitor',np.int32)]) 

#TODO: when you solve really large problems, will have to split up self.dict by tree
# to manage size of dictionary (so instead of one MergerTree class, wil have vector.)
class MergerTree:
    def __init__(self):
        self.data = []
        self.dict = {}
        self.trees = {}
        self.len   = 0

    def __getitem__(self, pid):
        return self.data[self.trees[pid]]
    
    def keys(self):
        return self.trees.keys()

    def items(self):
        return self.trees.items()

    def add_descendants(self, ge):
        idcs = np.flip(np.argsort(ge.Ns))
        NPs  = [(ge.Ns[i],ge.pids[i]) for i in idcs]
        Nnew = len(idcs)
        #link current entry to next one (main progenitor)
        #before moving on because main progenitor has same pid
        #so would otherwise move the dict reference and not allow
        #this entry to have its main progenitor updated.
        self.data[self.dict[ge.pid]]['main_progenitor'] = self.len
        self.data[self.dict[ge.pid]]['zf'] = ge.zf
        st = False
        for N,pid in NPs:
            el = np.array([(pid,N,-1,-1,-1)], dtype=MergerTreeNode)[0]
            self.data.append(el)#MergerTreeNode(pid,N,-1))
            if N > 1:
                self.dict[pid] = self.len 
                st = True
            self.len += 1
        if not st:
            del self.dict[ge.pid]
        for i in range(Nnew-1):
            self.data[-Nnew+i]['sibling'] = self.len-Nnew+i+1

    def add_tree(self, ge):
        el = np.array([(ge.pid,ge.N,ge.zf,-1,-1)], dtype=MergerTreeNode)[0]
        self.data.append(el)#MergerTreeNode(ge.pid,ge.N,ge.zf))
        self.trees[ge.pid] = self.len
        self.dict[ge.pid] = self.len
        self.len += 1
        self.add_descendants(ge)

    def print(self, tpid): #prints the tree of progenitors
        nd = self.data[self.trees[tpid]]
        visit_list = [(self.data[nd['main_progenitor']],1),]
        print("{:d}, N={:d}, zf={:.3f}".format(nd['pid'],nd['N'],nd['zf']))
        while(len(visit_list)>0):
            nd,depth = visit_list.pop()
            if nd['zf']>-1:
                zf_str = "zf:{:.3f}".format(nd['zf'])
            else:
                zf_str = ""
            print("  "*depth + "{:d}, N:{:d} ".format(nd['pid'], nd['N']) + zf_str)
            if nd['sibling'] != -1:
                visit_list.append((self.data[nd['sibling']], depth))
            if nd['main_progenitor']!= -1:
                visit_list.append((self.data[nd['main_progenitor']], depth+1))
    
    def check(self, tpid,tidx):
        if self.trees[tpid]<0 or self.trees[tpid]>=self.len:
            print("tidx_arg:{:d}".format(tidx),end=" ")
            print("tree indexing error: tpid={:d}, tidx={:d}, len:{:d}".format(tpid,self.trees[tpid],self.len),end=" ")
            assert False
        nd = self.data[self.trees[tpid]]
        visit_list = [(self.data[nd['main_progenitor']],nd['N'],True),]
        Nsinglets = 0
        scnt = nd['N']
        topN = nd['N']
        while(len(visit_list)>0):
            nd,lstN,prg = visit_list.pop()
            if prg:
                if scnt != topN:
                    print("\nscnt:{:d}, topN:{:d}".format(scnt,topN))
                    self.print(tpid)
                assert scnt == topN
                scnt = nd['N']
                topN = lstN
            else:
                scnt +=nd['N']
            if nd['N']==1:
                Nsinglets += 1
            if nd['main_progenitor'] != -1:
                visit_list.append((self.data[nd['main_progenitor']],nd['N'],True))
            if nd['sibling'] != -1:
                visit_list.append((self.data[nd['sibling']],None,False))
        if Nsinglets != self.data[self.trees[tpid]]['N']:
            print("\nNsinglets:{:d}, top MAP N:{:d}".format(Nsinglets, self.data[self.trees[tpid]]['N']))
            self.print(tpid)
        assert Nsinglets == self.data[self.trees[tpid]]['N']


    def largest_mainbranch_merger(self, tpid):
        nd = self.data[self.trees[tpid]]
        N = nd['N']
        time      = None
        masses    = None
        pids      = None
        depth     = None
        _depth    = 0
        max_frac  = -1
        while nd['main_progenitor'] != -1:
            pg = self.data[nd['main_progenitor']]
            _depth+=1
            if pg['N']>1:
                merger_Ns   = [pg['N'],]
                merger_pids = [pg['pid'],] 
                while (pg['sibling']!=-1) and (self.data[pg['sibling']]['N']>1):
                    pg = self.data[pg['sibling']]
                    merger_Ns.append(pg['N'])
                    merger_pids.append(pg['pid'])
                if (len(merger_Ns)>1) and (merger_Ns[1]/N>max_frac): 
                    time      = nd['zf']
                    masses    = merger_Ns
                    pids      = merger_pids 
                    depth     = _depth
                    max_frac  = merger_Ns[1]/N 
            nd = self.data[nd['main_progenitor']]
        
        return time,masses,pids,depth
   
    #traverse the tree getting formation statistics 
    def get_formation_stats(self, tpid):
        1+1

    #get accretion history
    def get_accretion_rates(self,tpid,z_bins):
        z_counts = []
        z = []
        mass = [] 
        root = self.data[self.trees[tpid]]
        last_descendant = self.trees[tpid]
        leftmost_sibling = self.trees[tpid]
        gained_mass = 0
        if self.data[root['main_progenitor']]['N']>1:
            visit_list = [(root['main_progenitor'],self.trees[tpid],None),]
            while len(visit_list)>0:
                # check node to see if corresponds to accretion
                nd,dcdt,lftsbl = visit_list.pop()
                if (lftsbl!=None) and (self.data[nd]['N']==1): #are iterating into a sibling
                    z.append(np.log(self.data[dcdt]['zf'])) #associate time of accretion with time of formation of descendant
                    mass.append(self.data[leftmost_sibling]['N'])
                    
                if lftsbl==None: #are entering main progenitor which will become the leftmost sibling of next objects
                    assert self.data[last_descendant]['N']==self.data[leftmost_sibling]['N']+gained_mass, \
                        "descendant:{:d}| lftsbl:{:d}, gained:{:d}".format(self.data[last_descendant]['N'], self.data[leftmost_sibling]['N'], gained_mass)
                    leftmost_sibling = nd
                    gained_mass = 0
                else:
                    gained_mass += self.data[nd]['N']
                
                # add next elements to iterate over
                if (self.data[nd]['main_progenitor'] != -1) and (self.data[self.data[nd]['main_progenitor']]['N']>1):
                    visit_list.append((self.data[nd]['main_progenitor'],nd,None))
                if self.data[nd]['sibling'] != -1:
                    visit_list.append((self.data[nd]['sibling'],dcdt,nd))

                last_descendant = dcdt
                    
            counts = np.histogram(z, bins=z_bins, weights=1.0/np.array(mass))[0]
            _z = (np.exp(z_bins[:-1])+np.exp(z_bins[1:]))/2.0
            sAR = counts / np.diff(z_bins) * (1+_z)/_z*H(_z) * 0.0010227121650456948 # to convert to 1/Gyr from km/s/Mpc
        else:
            sAR = np.zeros(len(z_bins)-1)
    
        return sAR
    
    # get statistics of all the mergers for plotting evolution of sim
    def get_merger_stats(self,tpid): 
        nd = self.data[self.trees[tpid]]
        visit_list = [(self.data[nd['main_progenitor']],nd['N'],True)]
        mrg_list = []
        mrgr = []
        while len(visit_list)>0:
            nd,lstN,prg = visit_list.pop()
            if prg:# you're done reading a set of siblings
                1+1 
            else: #you're iterating over siblings
                if nd['N']>1:
                    mrgr.append(nd['N'])
            if nd['main_progenitor'] != -1:
                visit_list.append((self.data['main_progenitor'],nd['N'],True))
            if nd['sibling'] != -1: 
                visit_list.append((self.data['sibling'],None,False))         
    
    #Get the time when the dominant branch started.
    #This will be formation time of a particular MAP.
    def mainbranch_formation_time(self, tpid):
        nd = self.data[self.trees[tpid]]
        while self.data[nd['main_progenitor']]['N']>1:
            nd = self.data[nd['main_progenitor']]
        return nd['zf']

    #get the time when the dominant branch got half of its mass
    def mainbranch_halfmass_time(self,tpid):
        nd = self.data[self.trees[tpid]]
        halfmass=0.5*nd['N']
        while (nd['main_progenitor'] != -1) and (self.data[nd['main_progenitor']]['N']>halfmass):
            nd = self.data[nd['main_progenitor']]
        return nd['zf']



def merge_progenitors(merger_tree, ges):
    Nge = len(ges)
    i=0
    while (i<Nge):
        if ges[i].pid in merger_tree.dict:
            merger_tree.add_descendants(ges[i])
            ges.pop(i)
            i -= 1 #now need to check this index again
            Nge -= 1 #shrink Nge to keep number of loop iterations constant
        i+=1
    
def get_merger_tree(parentdir, str_identifier, BoxSize, z_end=-1, recalc=False,RVPID=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    root = 0 #MPI.ROOT #root rank
    print("rank: {:d}".format(rank), end=" ")
    z = None
    subdirs = None
    if rank==root:
        z,subdirs = get_groupentry_subdirs(parentdir,z_end) 
    z = comm.bcast(z, root=root)
    subdirs = comm.bcast(subdirs, root=root)
    mg_dict_str = 'Images/merger_tree_dict_'+str_identifier+'.pkl'
    skip = None
    if rank==root:
        if os.path.isfile(mg_dict_str) and not recalc:
            print('Loading from file')
            fb =  open(mg_dict_str,'rb')
            final_merger_tree = pickle.loads(fb.read())
            fb.close()
            skip = True
        else:
            skip = False
    else:
        final_merger_tree = MergerTree()
    
    skip = comm.bcast(skip, root=root)
    if not skip:
        #build the merger tree file by file:
        merger_tree = MergerTree()

        for subdir_idx,subdir in enumerate(subdirs):
            filelist = None
            if rank==root:
                filelist = get_groupentry_filelist(subdir)
                if subdir_idx%20==0:
                    print("z:{:.3f}".format(z[subdir_idx]), end=" ")
                    
            filelist = comm.bcast(filelist, root=root)
            nfiles  = len(filelist)
            filechunks = np.full(comm_size, int(nfiles/comm_size))
            for r in range(comm_size):
                if r < nfiles%comm_size: #don't do usual trick of making rank 0 have less work because as it stands no ranks are preferred
                    filechunks[r] += 1
            fileidcs = np.zeros_like(filechunks)
            fileidcs[1:] = np.cumsum(filechunks[:-1])
            myfiles = filelist[fileidcs[rank]:fileidcs[rank]+filechunks[rank]]
            #print("slabs:", str(myfiles[0])[-4:],"-",str(myfiles[-1])[-4:], end= " ")
            incompl_ges = []
            if subdir_idx==0: #if first loop, want to avoid entire circular pass by initializing merger_tree
                for file in myfiles:
                    ges = get_ge_list(file,RVPID=RVPID)
                    for ge in ges:
                        merger_tree.add_tree(ge)
            else:
                for file in myfiles:
                    ges = get_ge_list(file,RVPID=RVPID)
                    merge_progenitors(merger_tree, ges) #modifies in place
                    incompl_ges.extend(ges)
        
            imdone = len(incompl_ges)==0
            done = comm.allreduce(imdone, op=MPI.LAND)
            npasses = 0
            while (not done) and (npasses<comm_size):
                #TODO: the dst/src should really be in an expanding alternating pattern;
                # this will essentially guarantee completion in two loop iterations because of 
                # the fact MAPs can't travel but so far.
                # Though this may not be as much of an optimization as it seems since all the new
                # MAPs will have to be passed around fully to know that they are actually new. Still
                # this would atleast decrease the message size.
                # (If really need speed up, be realistic and look no further than a few more than GroupRadius cells
                # to know is new.) 
                dst = (rank+1) % comm_size #rank to the right
                src = (rank-1+comm_size) % comm_size #rank to the left
                #pack manually
                Nge = len(incompl_ges)
                gepidv  = np.empty(Nge, dtype=np.uint64)
                geNv    = np.empty(Nge, dtype=np.int32)
                gezfv   = np.empty(Nge, dtype=np.float32)
                gesizes = np.empty(Nge, dtype=np.int32)
                for i,ge in enumerate(incompl_ges):
                    gepidv[i]  = ge.pid
                    geNv[i]    = ge.N
                    gezfv[i]   = ge.zf
                    gesizes[i] = len(ge.Ns)
                Nel = np.sum(gesizes)
                Ns   = np.empty(Nel, dtype=np.int32)
                pids = np.empty(Nel, dtype=np.uint64)
                wpos =0
                for i,sz in enumerate(gesizes):
                    Ns[wpos:wpos+sz] = incompl_ges[i].Ns
                    pids[wpos:wpos+sz] = incompl_ges[i].pids
                    wpos+=sz
                assert wpos==Nel
                #send each buffer
                msg_size = np.array([len(gepidv),len(Ns)], dtype=np.int32)
                comm.Sendrecv_replace(msg_size,dst,sendtag=0x59,source=src,recvtag=0x59) #get the buffer size you need to recieve
                #print("msg_size:[{:d},{:d}]".format(msg_size[0], msg_size[1]),end=" ")
                b1 = np.empty(msg_size[0], dtype=np.uint64)
                b2 = np.empty(msg_size[0], dtype=np.int32)
                b3 = np.empty(msg_size[0], dtype=np.float32)
                b4 = np.empty(msg_size[0], dtype=np.int32)
                b5 = np.empty(msg_size[1], dtype=np.int32)
                b6 = np.empty(msg_size[1], dtype=np.uint64)
                comm.Sendrecv(gepidv, dst, sendtag=0x1, recvbuf=b1, source=src, recvtag=0x1)
                comm.Sendrecv(geNv,   dst, sendtag=0x2, recvbuf=b2, source=src, recvtag=0x2)
                comm.Sendrecv(gezfv,  dst, sendtag=0x3, recvbuf=b3, source=src, recvtag=0x3)
                comm.Sendrecv(gesizes,dst, sendtag=0x4, recvbuf=b4, source=src, recvtag=0x4)
                comm.Sendrecv(Ns,     dst, sendtag=0x5, recvbuf=b5, source=src, recvtag=0x5)
                comm.Sendrecv(pids,   dst, sendtag=0x6, recvbuf=b6, source=src, recvtag=0x6)
                #unpack manually
                gepidv  = b1
                geNv    = b2
                gezfv   = b3
                gesizes = b4
                Ns      = b5
                pids    = b6
                incompl_ges = []
                wpos = 0
                for i,gepid in enumerate(gepidv): 
                    incompl_ges.append(GroupEntry(gepid,gezfv[i]))
                    incompl_ges[-1].N = geNv[i]
                    incompl_ges[-1].Ns   = Ns[wpos:wpos+gesizes[i]]
                    incompl_ges[-1].pids = pids[wpos:wpos+gesizes[i]]
                    wpos+=gesizes[i]
                
                assert wpos == msg_size[1]
                npasses += 1
                #print("npasses:{:d}, incompl_ges:{:d}".format(npasses, len(incompl_ges)), end=" ")
                if npasses<comm_size:
                    merge_progenitors(merger_tree, incompl_ges)
                imdone = len(incompl_ges)==0
                done = comm.allreduce(imdone, op=MPI.LAND)             

            #at this point if this process has anything in it's incompl_ges list these are new entries to add to merger tree
            if len(incompl_ges)>0:
                for ge in incompl_ges:
                    merger_tree.add_tree(ge)    

        #permute arrays into storage order for ease of merging
        print("permuting data", end=" ")
        sendcounts = comm.allgather(merger_tree.len)
        rankoffset = np.sum(sendcounts[:rank])
        idcs = np.zeros(merger_tree.len, dtype=np.int32)
        idx = 0 
        sibl = 0x1
        prog = 0x2
        for tpid,tidx in list(merger_tree.items()):
            merger_tree.check(tpid, tidx)
            merger_tree.trees[tpid] = idx + rankoffset #merger_tree tree dict must point to the start of this tree 
            swap_list = [(-1,-1,tidx)] # ptr_idx, sibling/pogenitor, myidx     
            while len(swap_list)>0:
                prevptr,ptrtyp,sch = swap_list.pop()
                idcs[idx] = sch
                sibling         = merger_tree.data[sch]['sibling']
                main_progenitor = merger_tree.data[sch]['main_progenitor']
                #update swap_list to visit the right next indices
                if sibling != -1:
                    if merger_tree.data[sibling]['N'] >1:
                        swap_list.append((sch,sibl,sibling))
                    else:
                        swap_list.insert(0,(sch,sibl,sibling))
                if main_progenitor != -1:
                    if merger_tree.data[main_progenitor]['N'] >1:
                        swap_list.append((sch,prog,main_progenitor))
                    else:
                        swap_list.insert(0,(sch,prog,main_progenitor))
                # Update nodes which point to this one.
                # This is where the structure of the tree is effectively encoded.
                if prevptr!=-1:
                    if ptrtyp == sibl:
                        merger_tree.data[prevptr]['sibling'] = idx + rankoffset
                    elif ptrtyp == prog:
                        merger_tree.data[prevptr]['main_progenitor'] = idx + rankoffset
                idx += 1
                if idx>idcs.size:
                    print("idx:{:d}".format(idx),end=" ")
                    merger_tree.print(tpid)

        assert idx==idcs.size
        data = [merger_tree.data[i] for i in idcs] 
        merger_tree.data = data
        print("done",end=" ") 
        #consolidate the trees in root
        print("sending trees to root",end=" ")
        send_sizes = comm.allgather(merger_tree.len)
        Nel = np.sum(send_sizes)
        ofsts = np.zeros_like(send_sizes, dtype=np.int32)
        ofsts[1:] = np.cumsum(send_sizes[:-1])
        merger_tree_data = None
        if rank==root:
            merger_tree_data = np.empty(Nel, dtype=MergerTreeNode)
        offsets= [MergerTreeNode.fields[field][1] for field in MergerTreeNode.names]
        types = [MPI.UINT64_T, MPI.INT32_T, MPI.FLOAT, MPI.INT32_T, MPI.INT32_T]
        block_lengths = [1,1,1,1,1]
        MergerTreeNode_MPI = MPI.Datatype.Create_struct(block_lengths, offsets, types) #MPI datatype for sending messages 
        MergerTreeNode_MPI.Commit()
        comm.Gatherv([np.array(merger_tree.data, dtype=MergerTreeNode), MergerTreeNode_MPI], [merger_tree_data,send_sizes,ofsts,MergerTreeNode_MPI],root=root)
        send_sizes = comm.allgather(len(merger_tree.trees))
        Nel = np.sum(send_sizes)
        ofsts = np.zeros_like(send_sizes)
        ofsts[1:] = np.cumsum(send_sizes[:-1])
        merger_tree_tpids=None
        merger_tree_tidxs=None
        if rank==root:
            merger_tree_tpids = np.empty(Nel, dtype=np.uint64)
            merger_tree_tidxs = np.empty(Nel, dtype=np.int32)
        comm.Gatherv(np.array(list(merger_tree.trees.keys()),dtype=np.uint64),  [merger_tree_tpids,send_sizes,ofsts,MPI.UINT64_T], root=root)
        comm.Gatherv(np.array(list(merger_tree.trees.values()),dtype=np.int32), [merger_tree_tidxs,send_sizes,ofsts,MPI.INT32_T], root=root)
        if rank==root:
            print("consolidating in root", end=" ")
            final_merger_tree = MergerTree()
            final_merger_tree.data = merger_tree_data  
            final_merger_tree.len = len(final_merger_tree.data)
            #copy over dictionary entries
            for i,tpid in enumerate(merger_tree_tpids):
                final_merger_tree.trees[tpid] = merger_tree_tidxs[i]
                final_merger_tree.check(tpid,merger_tree_tidxs[i])
            with open(mg_dict_str, 'wb') as fp:
                pickle.dump(final_merger_tree,fp)
            del merger_tree
        print("done", end=" ")

    
    return z,final_merger_tree

map_dtype = np.dtype([('m',np.int32), ('pos',np.float32,(3)), ('vel',np.float32,(3))])   
@nb.jit
def gen_map_struct(map_list,rv):
    cnt = 0
    map_pids   = np.empty((len(map_list)), dtype=np.uint64)
    map_struct = np.empty((len(map_list),7),dtype=np.float32) #for now, just over-allocate
    for i,m in enumerate(map_list):
        #tmp = np.empty(1, dtype=map_dtype).item()
        if m['m']>1:
            map_pids[cnt]       = m['pid']#np.empty(1, dtype=map_dtype).item()
            map_struct[cnt,0]   = m['m']
            map_struct[cnt,1:4] = rv[0][i,:]
            map_struct[cnt,4:7] = rv[1][i,:]
            cnt+=1
    map_pids   = map_pids[:cnt]
    map_struct = map_struct[:cnt,:]
    return map_pids, map_struct

def load_MAPData(parent_dir,BoxSize,str_identifier, zf=0.2,recalc=False, RVPID=False):
    map_struct_str = 'Images/map_struct_'+str_identifier+'_z={:.2f}.pkl'.format(zf)
    if os.path.isfile(map_struct_str) and not recalc:
        print('Loading from file')
        with open(map_struct_str,'rb') as fb:
           map_pids, map_struct, map_dict = pickle.loads(fb.read())
    else:
        #get closest file to desired redshift
        filedir = None
        prev_diff = 1e20 #just a number bigger than possible redshift difference
        prev_z = -1
        prev_filedir = None
        basedir = os.path.join(parent_dir, 'tracers/')
        for subdir in os.listdir(basedir):
            fn = os.path.join(os.path.join(basedir, subdir), 'TracersMAPData0000')
            if os.path.isfile(fn):
                res = re.search(r'Step(\d{4})_z(\d+\.\d+)', subdir)
                z = float(res.group(2))
                step = int(res.group(1))
                if(abs(z-zf)>prev_diff): 
                    if prev_diff!=0:
                        print("Closest redshift to desired occured at z={:.3f} during Step:{:d}".format(prev_z,step))
                    filedir = prev_filedir
                    break
                else:
                    prev_diff = abs(z-zf)
                    prev_z = z
                    prev_filedir = os.path.dirname(fn)

        #read map data
        if(filedir==None): #this means was last file (that contained TracersMAPData)
            filedir = Path(basedir) / 'Step{:04d}_z{:.3f}'.format(step,z)
        
        print("Reading from directory:", filedir)
        filelist   = get_groupentry_filelist(filedir, filebase="TracersMAPData")
        map_pids   = np.empty((0), dtype=np.uint64)
        map_struct = np.empty((0,7), dtype=float)
        for file in tqdm(filelist,desc='Generating map_dict...'):
            _maps = np.fromfile(file, dtype=MAPData_dtype)
            rv = bitpacked.unpack_rvint(_maps['rv'], BoxSize.value, float_dtype=np.float32)
            #for i,_m in enumerate(_maps):
            #    pv = bitpacked.unpack_rvint(_m['rv'],BoxSize,float_dtype=np.float32)
            #    maps[_m['pid']] = {'m':int(_m['m']),'pos':rv[0][i,:].squeeze(),'vel':rv[1][i,:].squeeze()}
            if len(_maps)>0:
              mp,ms = gen_map_struct(_maps, rv)
              if RVPID:
                unpackpids(mp) #inplace
              map_pids = np.concatenate((map_pids, mp), axis=0)
              map_struct = np.concatenate((map_struct,ms),axis=0)

        map_dict = {mpid:np.int32(i) for i,mpid in enumerate(map_pids) }
        print('Writing output to file')
        fp = open(map_struct_str,'wb')
        pickle.dump((map_pids,map_struct,map_dict), fp, protocol=2)
        fp.close()
    
    print('done')
    return (map_pids, map_struct), map_dict 
    
#accounts for fact that some halos wrap
@nb.njit
def _dist_from_com(pos,x_com,BoxSize,r100):
    diff = np.zeros((len(x_com),3))
    for dim in range(3):
        ds = np.abs(pos[dim]-x_com[:,dim])
        dp = BoxSize - ds
        dp[dp>r100] = np.inf
        diff[:,dim] = np.minimum(ds,dp)
    return np.sqrt(np.sum(diff**2,axis=1))

def Phi(h,x,m,mp,cat):
    phi = 0
    h_pos = cat.subsamples['pos'][h['npstartA']:h['npstartA'] + h['npoutA']]
    phi += np.sum(G.value*m*mp**2/np.linalg.norm(x-h_pos,axis=1))
    h_pos = cat.subsamples['pos'][h['npstartB']:h['npstartB'] + h['npoutB']]
    if h_pos.shape[0]>0 and h_pos.shape[1]>0:
        phi += np.sum(G.value*m*mp**2/np.linalg.norm(x-h_pos,axis=1))
    return phi

#create dictionary of halos pids to parent halo
@nb.jit(nopython=True,nogil=True)
def _add_pids(pid_dict,pids,ih):
    unpackpids(pids) #inplace
    for pid in pids:
        pid_dict[pid] = ih

@nb.jit(nopython=True,nogil=True)
def _compare(maps_tup,halo_pids,BoxSize):
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    Nmap = len(map_pids)
    map_pids_out = np.empty(Nmap, dtype=np.uint64)
    halo_ids     = np.empty(Nmap, dtype=np.int32)
    cnt = 0
    for im,map_pid in enumerate(map_pids):
        if map_pid in halo_pids: #consider this a match
            ih = halo_pids[map_pid]
            map_pids_out[cnt] = map_pid
            halo_ids[cnt] = ih
            cnt+=1
        
    return map_pids_out[:cnt], halo_ids[:cnt]

def make_halo_map_dict(halos_N, npstart, npout, halo_PID, maps_tup, BoxSize, Mp, zf, str_identifier, recalc=False):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    comm_size = comm.Get_size()
    root = 0 #root rank  
    hm_dict = dict()
    hm_dict_str = 'Images/hm_dict_'+str_identifier+'_z={:.2f}.pkl'.format(zf)
    if os.path.isfile(hm_dict_str) and (not recalc):
        if rank==root:

            #workaround for the fact that the pickling environment had a different name for illustris_utilities
            import illustris_utilities as illutil
            import sys
            sys.modules['illustris_utilities'] = illutil
            
            print('Loading from file', end=' ')
            fb =  open(hm_dict_str,'rb')
            hm_dict = pickle.loads(fb.read())
            fb.close()
    else:
        print(rank,end=": ")
        displs = None
        sendcounts = None
        subhalos = None
        #ParticleIDs   = None
        Nhalo=len(halos_N)
        if rank==root:
            print('Generating from sim outputs',end=' ')
            
        chunksize = int(Nhalo/comm_size)
        sendcounts = np.full(comm_size, chunksize)
        for i in range(1,Nhalo%comm_size+1):
            sendcounts[i]+=1
        displs = np.zeros_like(sendcounts)
        displs[1:] = np.cumsum(sendcounts[:-1]) 
        chunksize  = sendcounts[rank]
        halo_pids = Dict.empty(key_type=uint64, value_type=int32)
        cnt = 0
        for ih in range(chunksize):
            idx = displs[rank] + ih
            ParticleIDs = np.append(halo_PID[npstart[0,idx]:npstart[0,idx]+npout[0,idx]], halo_PID[npstart[1,idx]:npstart[1,idx]+npout[1,idx]])
            _add_pids(halo_pids, ParticleIDs, idx)
            cnt+=1
        if rank<comm_size-1:
            assert displs[rank]+cnt==displs[rank+1]
        else:
            assert displs[rank]+cnt==displs[rank]+sendcounts[rank]
            
        print("h_pids", end=" ")
        map_pids_, halo_ids_ = _compare(maps_tup,halo_pids,BoxSize)        
        Nel = len(map_pids_)
        sendcounts = comm.allgather(Nel)
        disps = np.zeros_like(sendcounts)
        disps[1:] = np.cumsum(sendcounts[:-1])
        print("Nel={:d}".format(Nel), end=" ")
        if rank==root:
            Nel = np.sum(sendcounts)
            map_pids   = np.empty(Nel,dtype=np.uint64)
            halo_ids   = np.empty(Nel,dtype=np.int32)
        else:
            map_pids = None
            halo_ids = None
        
        comm.Gatherv(map_pids_,    [map_pids, sendcounts, disps, MPI.UINT64_T], root=root) 
        comm.Gatherv(halo_ids_,    [halo_ids, sendcounts, disps, MPI.INT32_T],  root=root)
        if rank==root:
            for i,map_pid in enumerate(map_pids):
                ih = halo_ids[i]
                if ih in hm_dict:
                    hm_dict[ih].append(map_pid)
                else:
                    hm_dict[ih] = HaloMapRec(halos_N[ih],map_pid)
         
            print('Writing output to file',end=" ")
            fp = open(hm_dict_str,'wb')
            pickle.dump(hm_dict, fp, protocol=2)
            fp.close()
            print('done', end=" ")
    
    return hm_dict        



#halo_masses must be the list of halo masses(in original order) matching the struct that gm_dict
# was built comparing against.
def make_HOD_plots(hm_dict_use, maps_tup, map_dict, MinGalaxySize, str_identifier, Mp,Mp_tng, sel, BoxSize, zf, figsize=(12,9), cat={}, basePath="", snapno=-1, save=False, mthresh=(0,), MinMass=(0,), doTNG=False, ncol=2):
    #doTNG = (basePath!="") and (snapno>-1)
    fs = 18
    
    tng_halo_masses = il.groupcat.loadHalos(basePath,snapno,fields=['GroupMassType'])[sel,il.snapshot.partTypeNum('dm')]*1e10
    if doTNG:
        map_halo_masses = tng_halo_masses
    else:
        map_halo_masses = cat.halos['N']*Mp.value
        
    map_pids = maps_tup[0]
    map_struct = maps_tup[1]
    unmatched_halo_idcs = []
    matched_halo_idcs = []
    Nhalo_map = len(map_halo_masses)
    threshs = list(zip(mthresh,MinMass))
    Nplt = len(threshs)
    nrow = int(np.ceil(Nplt/ncol))
    fig,ax = plt.subplots(nrow,ncol,figsize=figsize,sharex=True, sharey=True,gridspec_kw={'wspace':0.,'hspace':0.05})
    fig.suptitle('z={:.2f}'.format(zf))
    ax = np.atleast_2d(ax)
    ax[-1,0].set_xlabel(r'$M_h \;\;[M_{\odot}/h]$', fontsize=fs)
    if ax.shape[1]>1:
        ax[-1,1].set_xlabel(r'$M_h \;\;[M_{\odot}/h]$', fontsize=fs)
    #ax[0,0].set_title(r"$\langle N_{cen}\rangle(M_h) \;\text{and}\; \langle N_{sat}\rangle(M_h)$")
    #ax[0,1].set_title(r"$\langle N_{cen}\rangle(M_h) \;\text{and}\; \langle N_{sat}\rangle(M_h)$")
    """
    ax0 = ax[0,0].secondary_xaxis('top', functions=(lambda N: N/Mp.value, lambda M: M*Mp.value))
    ax0.set_xlabel(r'Halo [N] (Abacus)')
    if ax.shape[1]>1:
        ax1 = ax[0,1].secondary_xaxis('top', functions=(lambda N: N/Mp.value, lambda M: M*Mp.value))
        ax1.set_xlabel(r'Halo [N] (Abacus)')
    """
    for nplt,(mthr,mm) in enumerate(threshs):
        row = int(nplt//ncol)
        col = int(nplt%ncol)
        mthr = mthr/Mp.value
        mm = mm/1e10

        satellites = np.zeros_like(map_halo_masses)
        centrals = np.zeros_like(map_halo_masses)
            
        for ih in range(Nhalo_map):
            if ih not in hm_dict_use:
                unmatched_halo_idcs.append(ih)
            else:
                mpids = hm_dict_use[ih].map_pids
                masses = map_struct[[map_dict[mp] for mp in mpids], 0]
                nmap = np.count_nonzero(masses>mthr)
                if nmap>1:
                    matched_halo_idcs.append(ih)
                    centrals[ih] = 1 
                    satellites[ih] = nmap-1
                elif nmap==1:
                    matched_halo_idcs.append(ih)
                    centrals[ih] = 1
                    satellites[ih] = 0
    
    
        unm_mass = map_halo_masses[unmatched_halo_idcs]
        m_mass = map_halo_masses[matched_halo_idcs]
        bins = np.logspace(np.log10(np.min(tng_halo_masses)), np.log10(np.max(tng_halo_masses)), 30)
    
        #get the Illustris central and satellites counts
        #if doTNG:         
        ptype = il.snapshot.partTypeNum('star')
        flag = il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloFlag'])\
                & (il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloMassType'])[:,ptype]>mm)
        gfs = il.groupcat.loadHalos(basePath,snapno,fields=['GroupFirstSub'])[sel]
        gns = il.groupcat.loadHalos(basePath,snapno,fields=['GroupNsubs'])[sel]
        centrals_ill = np.zeros_like(tng_halo_masses)
        satellites_ill = np.zeros_like(tng_halo_masses)
        for i in range(len(tng_halo_masses)):
            if gfs[i]>-1:
                msk = flag[gfs[i]:gfs[i]+gns[i]]
                centrals_ill[i] = msk[0]
                if gns[i]>1:
                    satellites_ill[i] = np.count_nonzero(msk[1:])
    
        #get expectation values in bins
        idcs_map = np.digitize(map_halo_masses, bins)-1    
        idcs_tng = np.digitize(tng_halo_masses, bins)-1
        
        bn = bins[0:-1] + np.diff(bins)/2.
        centrals_n     = np.zeros_like(bn)
        satellites_n   = np.zeros_like(bn)
        #if doTNG:
        centrals_ill_n     = np.zeros_like(bn)
        satellites_ill_n   = np.zeros_like(bn)
            
        for i in range(len(bins)-1):
            msk = i==idcs_map
            nnz = np.count_nonzero(msk)
            if nnz>0:
                centrals_n[i]     = np.mean(centrals[msk])
                satellites_n[i]   = np.mean(satellites[msk])
            msk = i==idcs_tng
            nnz = np.count_nonzero(msk)
            if nnz>0:
                #if doTNG:
                centrals_ill_n[i]     = np.mean(centrals_ill[msk])
                satellites_ill_n[i]   = np.mean(satellites_ill[msk])  
        
        # make plots
        lw=2.5
        ms=11.2
        #if doTNG:
        ax[row,col].plot(bn, centrals_ill_n,'--', lw=lw, ms=ms, c='indianred')#, label='central TNG')
        ax[row,col].plot(bn, satellites_ill_n,'-', lw=lw, ms=ms, c='indianred')#, label='satellites TNG') #indianred
        man,exp = get_sci_not(mm*1e10)
        ax[row,col].plot(bn, centrals_ill_n+satellites_ill_n,'r*--', lw=lw, ms=ms,  label=r'TNG')#$M^*_{{TNG}}>{:.2f}\times 10^{{{:d}}}\;\;M_{{\odot}}/h$'.format(man, exp))

        ax[row,col].plot(bn, centrals_n,':', lw=lw, ms=ms, c='tab:blue')#, label='central')
        ax[row,col].plot(bn, satellites_n,'-', lw=lw, ms=ms, c='tab:blue')#, label='satellites')
        man, exp = get_sci_not(mthr*Mp.value)
        ax[row,col].plot(bn, centrals_n+satellites_n, 'b.-',lw=lw, ms=ms, label=r'MAP')#'$M_{{MAP}}>{:.2f}\times 10^{{{:d}}}\;\;M_{{\odot}}/h$'.format(man,exp))

        #format plots
        ax[row,col].grid()
        ax[row,col].set(xscale='log',yscale='log' ,ylim=[9e-3,1.5e2], xlim=[1e11,1e15])
        ax[row,col].xaxis.set_tick_params(labelsize=18)
        ax[row,col].yaxis.set_tick_params(labelsize=18)
        ax[row,col].legend(loc='upper left')
            
        xlo,xhi = ax[row,col].get_xlim()
        ylo,yhi = ax[row,col].get_ylim()

        #if doTNG:
        man, exp = get_sci_not(np.sum(centrals_ill+satellites_ill)/BoxSize.value**3) 
        ax[row,col].text(xlo*(xhi/xlo)**0.025, ylo*(yhi/ylo)**0.72,r"$n={:.1f}\times 10^{{{:d}}}\;\; [Mpc/h]^{{-3}}$".format(man, exp))
        #ax[row,col].set_title(r"$n={:.1f}\times 10^{{{:d}}}\;\; [Mpc/h]^{{-3}}$".format(man, exp), fontsize=fs)
        
        M_central = np.interp(0.5, centrals_n, bn)
        print("M_central: {:.4e}".format(M_central))
        #ax[row,col].text(xlo*(xhi/xlo)**0.025, ylo*(yhi/ylo)**0.62, r"$M_{{central}}: {:.2e}$".format(M_central))
        
        #man, exp = get_sci_not(np.sum(centrals+satellites)/BoxSize.value**3)
        #ax[row,col].text(xlo*(xhi/xlo)**0.025, ylo*(yhi/ylo)**0.61,r"$n_{{MAP}}={:.1f}\times 10^{{{:d}}}\;\; [Mpc/h]^{{-3}}$".format(man, exp))
        if col==0:
            ax[row,col].set_ylabel(r'$\langle N\rangle$', fontsize=fs)
    
        print(np.sum(centrals)/np.sum(satellites)) 

    plt.show()

    #if doTNG:
    return centrals, satellites, centrals_ill, satellites_ill, fig
    #else:
    #    return centrals, satellites, [] , []


def halo_wise_corr(basePath, snapno, str_identifier, parent_dir,zf, NP,gal_map_dict, map_dict, maps_tup, Mp, Mp_tng, sel, BoxSize, type_str='gg', figsize=(14,12), mthresh=1e10, min_stellar_mass=1e5, save=False, doTNG=True):
    map_struct= maps_tup[1]
    dmtype = il.snapshot.partTypeNum('dm')
    stype  = il.snapshot.partTypeNum('star')
    tng_halo_masses = il.groupcat.loadHalos(basePath,snapno,fields=['GroupMassType'])[sel,dmtype]*1e10
    if doTNG:
        map_halo_masses = tng_halo_masses
    else:
        map_halo_masses = cat.halos['N']*Mp.value
    Nhalo_map = len(map_halo_masses)
    
    #get the central and satellite associations
    satellites = np.zeros_like(map_halo_masses)
    centrals = np.zeros_like(map_halo_masses)
        
    for ih in tqdm(range(Nhalo_map)):
        if ih in gal_map_dict:
            mpids = gal_map_dict[ih].map_pids
            masses = map_struct[[map_dict[mp] for mp in mpids], 0]
            nmap = np.count_nonzero(masses>mthresh/Mp.value)
            if nmap>1:
                centrals[ih] = 1 
                satellites[ih] = nmap-1
            elif nmap==1:
                centrals[ih] = 1
                satellites[ih] = 0
    stp=2
    nbins = 6
    bin_edges = np.logspace(np.log10(1e12), np.log10(np.max(tng_halo_masses)), nbins+1)
    bins = bin_edges[:-1]+np.diff(bin_edges)/2.
    #get the Illustris central and satellites counts
    #if doTNG:         
    ptype = il.snapshot.partTypeNum('star')
    flag = il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloFlag'])\
            & (il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloMassType'])[:,ptype]>min_stellar_mass/1e10)
    gfs = il.groupcat.loadHalos(basePath,snapno,fields=['GroupFirstSub'])[sel]
    gns = il.groupcat.loadHalos(basePath,snapno,fields=['GroupNsubs'])[sel]
    centrals_ill   = np.zeros_like(tng_halo_masses)
    satellites_ill = np.zeros_like(tng_halo_masses)
    for i in tqdm(range(len(tng_halo_masses))):
        if gfs[i]>-1:
            msk = flag[gfs[i]:gfs[i]+gns[i]]
            centrals_ill[i] = msk[0]
            if gns[i]>1:
                satellites_ill[i] = np.count_nonzero(msk[1:])

    #get expectation values in bins
    idcs_map = np.digitize(map_halo_masses, bin_edges)-1
    idcs_tng = np.digitize(tng_halo_masses, bin_edges)-1
        
    #make plots
    fig,ax = plt.subplots(int(np.ceil(len(bins)/stp)),2,figsize=figsize,sharex=True,gridspec_kw={'hspace':0.,'wspace':0.})
    _tng_pos = il.groupcat.loadSubhalos(basePath,snapno,fields=['SubhaloPos'])/1e3

    if type_str=='gg':
        rand_part = []
        rand_part_tng = []
    else:
        rand_part     = get_particle_pos(parent_dir,BoxSize,zf,NP,max_p=1e7)
        rand_part_tng = get_particle_pos_tng(basePath,snapno,'dm',max_p=1e7)/1e3
    
    min_size = 1e1
    rmin = 2e-2
    rmax = 30
    r_edges = np.logspace(np.log10(rmin), np.log10(rmax), 31)
    for i,bn in tqdm(enumerate(bins)):
        row = i//stp
        tng_pos_c, tng_pos_s = [], []
        map_pos_c, map_pos_s = [], []
        msk_map = i==idcs_map
        msk_tng = i==idcs_tng
        
        hidxs = np.flatnonzero(msk_map&(centrals>0))
        if len(hidxs>0):
            for ih in hidxs:
                _mi = np.array([map_dict[mpid] for mpid in gal_map_dict[ih].map_pids])
                i_cen = np.argmax(maps_tup[1][_mi,0])
                map_pos_c.append(maps_tup[1][_mi[i_cen],1:4])
                if satellites[ih]>0:
                    map_pos_s.extend(maps_tup[1][_mi[np.arange(len(_mi))!=i_cen],1:4])
        map_pos_s = np.vstack((map_pos_c, map_pos_s))


        tng_pos_c = _tng_pos[gfs[msk_tng&(centrals_ill>0)], :]
        subhalos = list(zip(gfs[msk_tng&(satellites_ill>0)],gns[msk_tng&(satellites_ill>0)]))
        for (fs,ns) in subhalos:
            idx = np.arange(fs+1,fs+ns)
            tng_pos_s.extend(_tng_pos[idx[flag[idx]],:])
        tng_pos_s = np.vstack((tng_pos_c, tng_pos_s))
        
        if len(map_pos_c)>min_size:
            r_mapc, zeta_mapc = get_xi(np.array(map_pos_c), r_edges, BoxSize, pos2=rand_part).get_corr(return_sep=True)
            ax[row,0].plot(r_mapc, r_mapc**2*zeta_mapc, '.-', color=colormaps['viridis'](i/len(bins)), 
                           label=r'$M_h={:.2e}M_{{\odot}}$'.format(bn))
        if len(map_pos_s)>min_size:
            r_maps, zeta_maps = get_xi(np.array(map_pos_s), r_edges, BoxSize, pos2=rand_part).get_corr(return_sep=True)
            ax[row,1].plot(r_maps, r_maps**2*zeta_maps, '.-', color=colormaps['plasma'](i/len(bins)), 
                           label=r'$M_h={:.2e}M_{{\odot}}$'.format(bn))
        if len(tng_pos_c)>min_size:
            r_tngc, zeta_tngc = get_xi(np.array(tng_pos_c, dtype=float), r_edges, BoxSize, pos2=rand_part_tng).get_corr(return_sep=True)
            ax[row,0].plot(r_tngc, r_tngc**2*zeta_tngc, '*--', color=colormaps['viridis'](i/len(bins)))
        if len(tng_pos_s)>min_size:
            r_tngs, zeta_tngs = get_xi(np.array(tng_pos_s, dtype=float), r_edges, BoxSize, pos2=rand_part_tng).get_corr(return_sep=True)
            ax[row,1].plot(r_tngs, r_tngs**2*zeta_tngs, '*--', color=colormaps['plasma'](i/len(bins)))

    ax[0,0].set(title=r'$r^2\xi_{{{:s}}}(r)$ centrals'.format(type_str))
    ax[0,1].set(title=r'$r^2\xi_{{{:s}}}(r)$ centrals & satellites'.format(type_str))
    ax[-1,0].set(xlabel=r'r [Mpc/h]', yscale='linear')
    ax[-1,1].set(xlabel=r'r [Mpc/h]', yscale='linear')#, ylim=[-2,2])
    for i in range(0,len(bins),stp):
        row = int(i//stp)
        ax[row,0].grid()
        ax[row,1].grid()
        ax[row,0].set(xscale='log')
        ax[row,1].set(xscale='log')
        ylim = ax[row,0].get_ylim()
        ax[row,0].text(rmin+(rmax-rmin)*0.01, ylim[0]+(ylim[1]-ylim[0])*0.9, 
                     r"[{:.2e} - {:.2e}] $[M_{{\odot}}]$".format(bin_edges[i],bin_edges[min(i+stp,nbins)]))
        ax[row,0].text(rmin+(rmax-rmin)*0.01, ylim[0]+(ylim[1]-ylim[0])*0.8, 
                     '[{:.2e} - {:.2e}][N]'.format(bin_edges[i]/Mp_tng.value, bin_edges[min(i+stp,nbins)]/Mp_tng.value))
    
    if save:
        fig.savefig(r'Images/halo_xi_'+str_identifier+'.png', bbox_inches='tight', transparent=True)
    plt.show()

def plot_corr(basePath,snapno,parent_dir, NP, zf, str_identifier,maps_tup, abundances, m_map, m_tng1, m_tng2, Mp, edges, BoxSize, figsize=(12,8), ncol=2, type_str='gm', save=False, binned=False):
    star_type= il.snapshot.partTypeNum('star')
    
    subhalos1 = il.groupcat.loadSubhalos('/mnt/hardraid_outer/ajohnson/TNG300/TNG300-1/output', snapno, fields=['SubhaloPos','SubhaloFlag','SubhaloMassType'])
    subhalos2 = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloPos','SubhaloFlag','SubhaloMassType'])

    map_threshs  = np.empty_like(abundances)
    tng_threshs1 = np.empty_like(abundances)
    tng_threshs2 = np.empty_like(abundances)
    for i,n in enumerate(abundances):
        N = int(n*BoxSize.value**3)
        map_threshs[i] = m_map[-N]
        tng_threshs1[i] = m_tng1[-N]
        tng_threshs2[i] = m_tng2[-N]   
    threshs = list(zip(np.atleast_1d(map_threshs), np.atleast_1d(tng_threshs1), np.atleast_1d(tng_threshs2)))
    
    SubhaloPos1 = np.array(subhalos1['SubhaloPos'],dtype=float)/1000
    SubhaloPos2 = np.array(subhalos2['SubhaloPos'],dtype=float)/1000
    
    Nplt = len(threshs)
    nr = int(np.ceil(Nplt/ncol))
    fig,ax = plt.subplots(nr,ncol,figsize=figsize,sharex=True, gridspec_kw={'hspace':0.1,'wspace':0.12})
    ax = np.atleast_2d(ax)
    #ax.plot(r_abacus,   r_abacus**2*xi_abacus,'-', lw=8,c='tab:gray')#, label=r'DM (MAP)')
    #ax.plot(r_tng,r_tng**2*xi_tng,'-', lw=4,c='dimgray')#, label=r'DM (TNG)')
    if type_str=='gm':
        rand_part     = get_particle_pos(parent_dir,BoxSize,zf,NP,max_p=1e7)
        rand_part_tng2 = get_particle_pos_tng(basePath,snapno,'dm',max_p=1e7)/1e3
        
    cnt = 0
    bias = np.empty((Nplt,len(edges)-1))
        
    for map_thresh, tng_thresh1, tng_thresh2 in tqdm(threshs[:len(threshs)-int(binned)]):
        row = int(cnt//ncol)
        col = int(cnt%ncol)
        
        #MAPs
        msk = maps_tup[1][:,0] >= map_thresh/Mp.value
        if binned:
            msk = msk & (maps_tup[1][:,0] < threshs[cnt+1][0]/Mp.value)

        r_gg, xi_gg = get_xi(maps_tup[1][msk,1:4], edges, BoxSize).get_corr(return_sep='True')
        if type_str=='gm':
            r_gm, xi_gm = get_xi(maps_tup[1][msk,1:4], edges, BoxSize, pos2=rand_part).get_corr(return_sep='True')
        
        #TNG
        msk1 = subhalos1['SubhaloFlag']\
            & (subhalos1['SubhaloMassType'][:,star_type]>=tng_thresh1/1e10)
        if binned:
            msk1 = msk1 & (subhalos1['SubhaloMassType'][:,star_type]<threshs[cnt+1][1]/1e10)
        r_tng_gg1, xi_tng_gg1 = get_xi(SubhaloPos1[msk1,:], edges, BoxSize).get_corr(return_sep='True')
        #if type_str=='gm':
        #    r_tng_gm1, xi_tng_gm1 = get_xi(SubhaloPos1[msk1,:], edges, BoxSize, pos2=rand_part_tng1).get_corr(return_sep='True')

        msk2 = subhalos2['SubhaloFlag']\
            & (subhalos2['SubhaloMassType'][:,star_type]>=tng_thresh2/1e10)
        if binned:
            msk2 = msk2 & (subhalos2['SubhaloMassType'][:,star_type]<threshs[cnt+1][2]/1e10)
        r_tng_gg2, xi_tng_gg2 = get_xi(SubhaloPos2[msk2,:], edges, BoxSize).get_corr(return_sep='True')
        if type_str=='gm':
            r_tng_gm2, xi_tng_gm2 = get_xi(SubhaloPos2[msk2,:], edges, BoxSize, pos2=rand_part_tng2).get_corr(return_sep='True')
        
        #get best fit power-law scaling
        """
        ydata = np.log(np.hstack((xi_gg, xi_tng_gg1)))
        xdata = np.log(np.hstack((r_gg,  r_tng_gg1)))
        nanmsk = ~np.isnan(ydata)
        popt, pcov = curve_fit(lambda x, a,b: a*x+b, xdata[nanmsk], ydata[nanmsk])
        """
        p = 2 #-popt[0]
        #if type_str=='gm':
        #    ydata = np.log(np.hstack((xi_gm, xi_tng_gm2)))
        #    xdata = np.log(np.hstack((r_gm,  r_tng_gm2)))
        #    nanmsk = ~np.isnan(ydata)
        #    popt, pcov = curve_fit(lambda x, a,b: a*x+b, xdata[nanmsk], ydata[nanmsk])
        #    p = -popt[0]
        
        #make plots
        man, exp = get_sci_not(tng_thresh1)
        ax[row,col].plot(r_tng_gg1, r_tng_gg1**p*xi_tng_gg1,'-.', lw=3, ms=11, c='darkviolet',label= r'$\xi_{gg}$ TNG300-1')
            #r'$M^*_{{TNG}}\geq{:.2f}\times 10^{{{:d}}}\;\;M_{{\odot}}/h$'.format(man, exp)
        #if type_str=='gm':
        #    ax[row,col].plot(r_tng_gm1, r_tng_gm1**p*xi_tng_gm1,'--', lw=3, ms=11, c='darkviolet', label=r'$\xi_{gm} TNG300-1$')

        man, exp = get_sci_not(tng_thresh2)
        ax[row,col].plot(r_tng_gg2, r_tng_gg2**p*xi_tng_gg2,'-', lw=3, ms=11, c='red',label= r'$\xi_{gg}$ TNG300-2')
            #r'$M^*_{{TNG}}\geq{:.2f}\times 10^{{{:d}}}\;\;M_{{\odot}}/h$'.format(man, exp))
        if type_str=='gm':
            ax[row,col].plot(r_tng_gm2, r_tng_gm2**p*xi_tng_gm2,'*-', lw=3, ms=11, c='red', label=r'$\xi_{gm}$ TNG300-2')
                             
        man, exp = get_sci_not(map_thresh)
        ax[row,col].plot(r_gg, r_gg**p*xi_gg,'-',c='blue', lw=3, ms=11,  label=r'$\xi_{gg}$ MAP')#r'$M_{{MAP}}\geq{:.2f}\times 10^{{{:d}}}\;\;M_{{\odot}}/h$'.format(man, exp))
        if type_str=='gm':
            ax[row,col].plot(r_gm, r_gm**p*xi_gm,'.-',c='blue', lw=3, ms=11, label=r'$\xi_{gm}$ MAP')

        bias[cnt,:] = np.sqrt(np.interp(r_tng_gg1,r_gg,xi_gg) / xi_tng_gg1)
        
        ax[row,col].set(xscale='log', yscale='linear')#, xlabel=r'r [Mpc/h]', title=r'$r^2\xi_{{{:s}}}(r)$'.format(type_str))
        ax[row,col].grid()
        #ax[row,col].legend()
        xlim = ax[row,col].get_xlim()
        ylim = ax[row,col].get_ylim()
        ax[row,col].set_ylim([0, ylim[1]])
        
        man, exp = get_sci_not(np.count_nonzero(msk1)/BoxSize.value**3)
        ax[row,col].text((xlim[1]/xlim[0])**0.01 *xlim[0], 0.11*ylim[1], r'p={:.2f}'.format(p))
        ax[row,col].text((xlim[1]/xlim[0])**0.01 *xlim[0], 0.02*ylim[1], r'$n={:.1f}\times 10^{{{:d}}}\;\; [h^{{3}} Mpc^{{-3}}]$'.format(man,exp))
        cnt+=1

        if row==nr-1:
            ax[row,col].set_xlabel(r'$r \;\;[h^{-1} Mpc]$')

    fig.suptitle(r'$r^p\xi(r)$, z={:.2f}'.format(zf))
    #ax[0,0].set_title(r'$r^2\xi_{{{:s}}}(r)$'.format(type_str))
    #ax[0,1].set_title(r'$r^2\xi_{{{:s}}}(r)$'.format(type_str))
    ax[0,-1].legend(loc='upper right')
    
    #handles, labels = ax[0,0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc='upper right')
    plt.show()
    if save:
        fig.savefig('Images/xi_{:s}_'.format(type_str)+str_identifier+'_z={:.2f}.png'.format(zf), bbox_inches='tight', transparent=True)
    
    return bias, r_tng_gg1
    
def get_cathalo_rvpid(hidx, cat, key='pos'):
    #h = cat.halos[cat.halos['id'].data==packpid(hid)]
    h  =cat.halos[hidx]
    if key=='pos' or key=='vel':
        res = np.empty((int(h['npoutA']+h['npoutB']),3))
    elif key=='pid':
        res = np.empty((int(h['npoutA']+h['npoutB'])))
    res[:h['npoutA']] = cat.subsamples[key][h['npstartA']:h['npstartA']+h['npoutA']]
    if h['npoutB']>0:
        res[h['npoutA']:h['npoutA']+h['npoutB']] = cat.subsamples[key][h['npstartB']:h['npstartB']+h['npoutB']]
    
    return res 
   
def plot_sample_halos(maps_tup, map_dict, hm_dict, BoxSize, str_identifier, Mp, Mp_tng, map_thresh=1, tng_thresh=1, cat={}, basePath="", snapno=-1, rand=False, n1=3,n2=3, midx=[], percentile=0.8,
                      figsize=(16,14),save=False):
    #plot a few halos and their containing maps
    map_pids   = maps_tup[0]
    map_struct = maps_tup[1]
    halo_masses = np.array([hd.N for hd in hm_dict.values()])
    if (basePath!="") != (snapno>-1):
        print("If you specify basePath, must specify snapno (and vice-versa)")
        assert False
    doTNG = (basePath!="") and (snapno>-1)
    if doTNG:
        star_type = il.snapshot.partTypeNum('star')
        dmtype   = il.snapshot.partTypeNum('dm')
        halos    = il.groupcat.loadHalos(basePath,snapno, fields=['GroupFirstSub','GroupNsubs', 'GroupPos'])
        subhalos = il.groupcat.loadSubhalos(basePath, snapno,fields=['SubhaloPos','SubhaloLenType','SubhaloFlag'])
        if cat!={}:
            print("Can't do both Compaso and TNG halos; hm_dict can only be associated with one")
        assert cat=={}
    if rand==True:
        ridx = np.arange(len(halo_masses))[np.random.randint(0,high=len(halo_masses),size=int(n1*n2))]
    elif len(midx)>=1:
        idx = midx
    else:
        l = int(len(halo_masses)*percentile-n1*n2/2)
        idx = np.argsort(halo_masses)[int(l-n1*n2/2):int(l+n1*n2/2)]
    
    print(idx)
    
    hds = [(k,hd) for k,hd in hm_dict.items()]
    fig,ax = plt.subplots(n1,n2,figsize=figsize,gridspec_kw={'wspace':0.22, 'hspace':0.12})
    ax=np.atleast_2d(ax)
    for p,ii in enumerate(idx):
        hid,hd = hds[ii]
        row = p//n2
        col = p%n2
        if doTNG:
            pos    = il.snapshot.loadHalo(basePath,snapno,hid,il.snapshot.partTypeNum('dm'),fields=['Coordinates'])[:50000]/1000
            subpos = subhalos['SubhaloPos'][halos['GroupFirstSub'][hid]:halos['GroupFirstSub'][hid]+halos['GroupNsubs'][hid],:]/1000
            subm   = subhalos['SubhaloLenType'][halos['GroupFirstSub'][hid]:halos['GroupFirstSub'][hid]+halos['GroupNsubs'][hid],star_type]
            msk    = subhalos['SubhaloFlag'][halos['GroupFirstSub'][hid]:halos['GroupFirstSub'][hid]+halos['GroupNsubs'][hid]]==1 & (subm>=(tng_thresh/Mp_tng.value))
            subpos = subpos[msk]
            subm   = subm[msk]
        elif cat!={}:
            pos = get_cathalo_rvpid(hid,cat,key='pos')
        else:
            print("Need to specify halo catalog")
            assert False
        m_idxs = np.array([map_dict[_i] for _i in hd.map_pids])
        MAP_m   = map_struct[m_idxs,0]
        MAP_pos = np.copy(map_struct[m_idxs,1:4])
        MAP_pos = MAP_pos[MAP_m>map_thresh/Mp.value,:]
        MAP_m   = MAP_m[MAP_m>map_thresh/Mp.value]
        if doTNG:
            MAP_pos = (MAP_pos + BoxSize.value) % BoxSize.value
        ax[row,col].set_facecolor('k')
        
        xyz = pos[:,:2].T
        p_est = gaussian_kde(xyz)
        c = p_est(xyz)
        ax[row,col].scatter(pos[:,0], pos[:,1], c=colormaps['magma'](0.1+0.9*c/max(c)))
        
        if(len(MAP_pos)>0):
            if doTNG and (subpos.shape[0]>0):
                #ax[row,col].scatter(halos['GroupPos'][hid][0]/1000, halos['GroupPos'][hid][1]/1000,marker='X',c='g', s=60,label='group pos')
                ax[row,col].scatter(subpos[:,0], subpos[:,1], marker='*', s=subm/np.max(subm)*220,
                            c=colormaps['summer_r'](0.1+subm/np.max(subm)),label='TNG galaxy')

            ax[row,col].scatter(MAP_pos[:,0], MAP_pos[:,1], marker='X',s=MAP_m/np.max(MAP_m)*220,
                            c=colormaps['Blues'](MAP_m/np.max(MAP_m)),label='MAPs')
        ax[row,col].grid()
        xplt = (ax[row,col].get_xlim()[1] - ax[row,col].get_xlim()[0])*0.01 + ax[row,col].get_xlim()[0]
        yplt = (ax[row,col].get_ylim()[1] - ax[row,col].get_ylim()[0])*0.95 + ax[row,col].get_ylim()[0]
        exp = int(np.floor(np.log10(halo_masses[ii]*Mp.value)))
        ax[row,col].text(xplt, yplt, r'$M_{{h}}={:.2f}\times 10^{{{:d}}}\;M_{{\odot}}/h$'.format(halo_masses[ii]*Mp.value/10**exp, exp),  c='w',clip_on=True)

        if col==0:
            ax[row,col].set(ylabel='Mpc/h')
        if row==n2-1:
            ax[row,col].set(xlabel='Mpc/h')
    
    handles,labels = [], []
    for h,l in zip(*ax[n1-1,n2-1].get_legend_handles_labels()):
        handles.append(h)
        labels.append(l)
    #fig.legend(handles,labels)
    plt.show() 
    if(save):
        fig.savefig('Images/Halos_'+str_identifier+'.png',bbox_inches='tight')
    
    
def get_box_msk(pos,ylim,zlim,xlo,xhi):
    return (pos[:,2]>zlim[0])  & (pos[:,2]<zlim[1]) &\
            (pos[:,1]>ylim[0]) & (pos[:,1]<ylim[1]) &\
            (pos[:,0]>xlo)     & (pos[:,0]<xhi)




def get_compaso_pos(cat,maps,NP,BoxSize,Nslab,z,mthresh=1,max_h=1e6,max_p=5e6):
    #halo particles
    halo_particle_pos = []
    #halo_particle_vel = []
    Nhalo = len(cat.halos)
    if Nhalo > max_h:
        idxs = rng.choice(Nhalo, size=int(max_h),replace=False)
    else:
        idxs = np.arange(Nhalo)
    x_L2com = np.empty((len(idxs),3), dtype=float)
    x_L2N   = np.empty((len(idxs),3), dtype=float)
    Nhalo_p = len(cat.subsamples['pos'])
    cnt = 0
    if Nhalo_p > max_p:
        dwnsmpl = max_p/Nhalo_p
        nwritten = 0
        for idx in idxs: #for idx in idxs:
            h = cat.halos[idx]
            n_2_read = int(dwnsmpl*h['npoutA'])
            p_idcs = rng.choice(h['npoutA'], size=n_2_read,replace=False)
            nwritten += n_2_read
            halo_particle_pos.extend(cat.subsamples['pos'][h['npstartA']:h['npstartA']+h['npoutA']].data[p_idcs,:])
            #halo_particle_vel.extend(cat.subsamples['vel'][h['npstartA']:h['npstartA']+h['npoutA']].data[p_idcs,:])
            if h['npoutB']>0:
                n_2_read = int(dwnsmpl*h['npoutB'])
                p_idcs = rng.choice(h['npoutB'], size=n_2_read,replace=False)
                nwritten += n_2_read
                halo_particle_pos.extend(cat.subsamples['pos'][h['npstartB']:h['npstartB']+h['npoutB']].data[p_idcs,:])
                #halo_particle_vel.extend(cat.subsamples['vel'][h['npstartB']:h['npstartB']+h['npoutB']].data[p_idcs,:])
            x_L2com[cnt,:] = h['x_L2com']
            x_L2N[cnt,:]   = h['L2_N'][0]
            cnt += 1
        halo_particle_pos = np.array(halo_particle_pos, dtype=np.double)[:nwritten]
        #halo_particle_vel = np.array(halo_particle_vel, dtype=np.double)[:nwritten]
    else:
        for idx in idxs: #for idx in idxs:
            h = cat.halos[idx]
            halo_particle_pos.extend(cat.subsamples['pos'][h['npstartA']:h['npstartA']+h['npoutA']].data)
            #halo_particle_vel.extend(cat.subsamples['vel'][h['npstartA']:h['npstartA']+h['npoutA']].data)
            if h['npoutB']>0:
                halo_particle_pos.extend(cat.subsamples['pos'][h['npstartB']:h['npstartB']+h['npoutB']].data)
                #halo_particle_vel.extend(cat.subsamples['vel'][h['npstartB']:h['npstartB']+h['npoutB']].data)
            x_L2com[cnt,:] = h['x_L2com']
            x_L2N[cnt,:]   = h['L2_N'][0]
            cnt += 1
        halo_particle_pos = np.array(halo_particle_pos, dtype=np.double)
        #halo_particle_vel = np.array(halo_particle_vel, dtype=np.double)
        
    return x_L2com, x_L2N, halo_particle_pos #, halo_particle_vel


def get_particle_pos(parent_dir,BoxSize,zf,NP,max_p=5e8, wrap_to_positive=False, return_vel=False):
    # get closest file to desired redshift
    filedir = None
    prev_diff = 1e20 #just a number bigger than possible redshift difference
    prev_z = -1
    prev_filedir = None
    for subdir in os.listdir(parent_dir):
        fn = os.path.join(parent_dir, subdir)
        res = re.search(r'slice(\d+\.\d+)', fn)
        if res:
            z = float(res.group(1))
            if(abs(z-zf)>prev_diff):
                if prev_diff!=0:
                    print("Closest redshift to desired occured at z={:.3f}".format(prev_z))
                z = prev_z
                break
            else:
                prev_diff = abs(z-zf)
                prev_z = z
                
    # get partilces               
    pos = np.empty((int(min(max_p,NP)),3),dtype=float)
    if return_vel:
        vel = np.empty((int(min(max_p,NP)),3),dtype=float)
    nwritten = 0
    sim_name = parent_dir.split("/")[-1]
    downsample = max_p/NP
    slicedir = Path(parent_dir) / "slice{:.03f}".format(z)
    #slicedir = Path(parent_dir) / Path("slices") / "z{:.03f}".format(z) / "field_pack9"
    for file in os.listdir(slicedir):
        if re.search(r"{:s}\.z{:.03f}\.slab\d*\.(field|L0)_pack9\.dat".format(sim_name,z), file):
            #tbl = read_abacus.read_pack9(str(slicedir/file),  format=r'pack9', return_vel=return_vel, return_header=False, return_pid=False)
            tbl = read_abacus.read_pack9(str(slicedir/file), return_vel=return_vel, return_header=False, return_pid=False)
            np_in_slab = len(tbl)
            if downsample<1:
                np_2_read = int(np.floor(np_in_slab*downsample))
                idxs = rng.choice(np_in_slab, size=np_2_read,replace=False)
                pos[nwritten:nwritten+np_2_read,:] = tbl['pos'][idxs].data
                if return_vel:
                    vel[nwritten:nwritten+np_2_read,:] = tbl['vel'][idxs].data
            else:
                np_2_read = np_in_slab
                pos[nwritten:nwritten+np_2_read,:] = tbl['pos'].data
                if return_vel:
                    vel[nwritten:nwritten+np_2_read,:] = tbl['vel'].data
            nwritten += np_2_read
            
    pos = pos[:nwritten,:]*BoxSize.value #remove any unused entries due to max_p/NP being floored
    if wrap_to_positive: #This is usually used for correlation functions, so wrap to positive
        pos =  (pos + BoxSize.value) % BoxSize.value
    
    if return_vel:
        vel = vel[:nwritten,:]
        return pos, vel
    else:
        return pos

def get_lc_pos(lcdir, BoxSize,map_dict, mthresh=1,max_p=5e8, wrap_to_positive=True):
    #get directory size to pre-allocate arrays
    filesize = 0
    for lcf in os.listdir(lcdir+'/rv'):
        if re.search(r".*LightCone0_rv_Step(\d{4}).asdf$", lcf):
            filesize += np.ceil(os.path.getsize(lcdir+'/rv/'+lcf)/4.) #4=sizeof(float)
    filesize = int(filesize)
    pos = np.empty((filesize,3))
    pid = np.empty(filesize)
    z = np.empty(filesize)
    msk = np.empty(filesize,dtype=bool)
    cnt = 0
    for lcf in os.listdir(lcdir+'/rv'):
        res = re.search(r".*LightCone0_rv_Step(\d{4}).asdf$", lcf)
        if res:
            step = int(res.group(1))
            with open(lcdir+'../lightcone/Step{:04d}/header'.format(step),'r') as header:
                line =header.readline()
                while line:
                    res = re.search(r'[^\w]Redshift\s*=\s*(\d*\.{,1}\d*e{,1}[+|-]{,1}\d*)', line)
                    if res:
                        _z = res.group(1)
                        break
                    line=header.readline()
            lc_pid = read_abacus.read_asdf(lcdir+'/pid/LightCone0_pid_Step{:04d}.asdf'.format(step))['pid']
            Nel = len(lc_pid)
            pid[cnt:cnt+Nel] = lc_pid
            pos[cnt:cnt+Nel,:] = read_abacus.read_asdf(lcdir+'/rv/'+lcf, colname='rvint')['pos']
            msk[cnt:cnt+Nel] = [unpackpid(lpid) in map_dict for lpid in lc_pid]
            z[cnt:cnt+Nel] = np.full(Nel,_z)
            cnt += Nel
    return pos[:cnt,:], pid[:cnt], msk[:cnt], z[:cnt]
    


def radial_density_profiles(cat, map_dict, maps_tup, hm_dict, Mp, basePath, snapno, sel, gal_map_dict, H0, Omega_M, 
                            halo_mass_bins = [(10,11.8), (12.8,13.0), (13.2,13.6), (13.6,14.2)], 
                            nbins=15, rmin=3e-2, rmax=1, map_thresh=3.7e11, tng_thresh=1.5e9, nrow=2,ncol=2,figsize=(13,9)):
    star_type = il.snapshot.partTypeNum('stars')
    dm_type   = il.snapshot.partTypeNum('dm') 
    hids, halo_masses = zip(*[ (key,hmr.N) for key,hmr in hm_dict.items()])
    hids = np.array(hids)
    halo_masses = np.log10(np.array(halo_masses)*Mp.value)
    fig, ax = plt.subplots(nrow, ncol, figsize=figsize, gridspec_kw={'wspace':0., 'hspace':0.}, sharex=True, sharey=True)
    ax = np.atleast_2d(ax)
    #fig.supylabel(r'$[h^3Mpc^{-3}]$')
    rhist_edges = np.logspace(np.log10(rmin), np.log10(rmax), nbins+1)
    #rhist_edges = np.linspace(2e-2, rmax, nbins+1)
    rhist = rhist_edges[:-1] + np.diff(rhist_edges)/2. 
    tng_halos    = il.groupcat.loadHalos(basePath, snapno, fields=['GroupMassType','GroupNsubs','GroupFirstSub', 'GroupPos','GroupCM'])
    tng_subhalos = il.groupcat.loadSubhalos(basePath, snapno, fields=['SubhaloMassType','SubhaloFlag', 'SubhaloPos'])
    for row in range(nrow):
        for col in range(ncol):

            h_mass_range = halo_mass_bins[row*ncol+col]
            _hids = hids[(halo_masses>=h_mass_range[0]) & (halo_masses<h_mass_range[1])]
            num_halos = len(_hids) #number of halos in this range
            
            rho_map = np.zeros(nbins)
            n_map = np.zeros(nbins)
            n_map_avg = np.array([0,0])
            print(r'Mass range:[10^{:.1f} - 10^{:.1f}], Num_halos:{:.0f}'.format(h_mass_range[0],h_mass_range[1],num_halos))
            for hid in _hids:
                halo = cat.halos[hid]
                #rmax = halo['rvcirc_max_com'] #halo['r25_com']*2.5
                
                # get halo radial profile
                #hpos = cat.subsamples['pos'][int(halo['npstartA']):int(halo['npstartA'] + halo['npoutA'])].data.reshape(-1,3)
                #if cat.subsamples['pos'][int(halo['npstartB']):int(halo['npstartB'] + halo['npoutB'])].shape[0]>0:
                #    hpos = np.extend(hpos, cat.subsamples['pos'][int(halo['npstartB']):int(halo['npstartB'] + halo['npoutB'])].data.reshape(-1,3))
                #r = np.linalg.norm(hpos-halo['x_com'].reshape(-1,3),axis=1)/rmax
                #rhonfw += np.histogram(r, bins=rhist_edges, density=True)[0]
                # 
                
                # get map radial profile
                m_ids = [map_dict[mpid] for mpid in hm_dict[hid].map_pids] 
                r = np.linalg.norm(maps_tup[1][m_ids,1:4] - halo['x_L2com'],axis=1)
                m = maps_tup[1][m_ids,0]*Mp.value
                p_hist_counts = np.histogram(r[m>map_thresh], bins=rhist_edges, density=True, weights=m[m>map_thresh])[0]
                n_hist_counts = np.histogram(r[m>map_thresh], bins=rhist_edges, density=True)[0]
                if(np.any(np.isnan(p_hist_counts))):
                    continue
                else:
                    rho_map += p_hist_counts
                    n_map_avg[0] += 1 #np.sum(p_hist_counts)
                if(np.any(np.isnan(n_hist_counts))):
                    continue
                else:
                    n_map += n_hist_counts
                    n_map_avg[1] += 1 #np.sum(n_hist_counts)
                    
            n_tng = np.zeros(nbins)
            rho_tng = np.zeros(nbins)
            n_tng_avg = np.array([0,0])
            tng_idcs = np.argwhere((np.log10(tng_halos['GroupMassType'][:,dm_type])+10>=h_mass_range[0]) & (np.log10(tng_halos['GroupMassType'][:,dm_type])+10<h_mass_range[1]))
            print("N tng halos: {:d}".format(np.count_nonzero(tng_idcs)))
            for idx in tng_idcs:
                idx = idx[0]
                # get tng radial profile
                start = tng_halos['GroupFirstSub'][idx]
                nel   =  tng_halos['GroupNsubs'][idx]
                tng_radii = np.linalg.norm(tng_subhalos['SubhaloPos'][start:start+nel,:] - tng_halos['GroupCM'][idx], axis=1)/1e3
                subsizes = tng_subhalos['SubhaloMassType'][start:start + nel, star_type]
                msk = tng_subhalos['SubhaloFlag'][start:start + nel] & (subsizes>tng_thresh/1e10)
                p_hist_counts = np.histogram(tng_radii[msk], bins=rhist_edges, density=True, weights=subsizes[msk])[0]
                n_hist_counts = np.histogram(tng_radii[msk], bins=rhist_edges, density=True)[0]
                if(np.any(np.isnan(p_hist_counts))):
                    continue
                else:
                    rho_tng += p_hist_counts
                    n_tng_avg[0] += 1 #np.sum(p_hist_counts)
                if(np.any(np.isnan(n_hist_counts))):
                    continue
                else:
                    n_tng += n_hist_counts
                    n_tng_avg[1] += 1 #np.sum(n_hist_counts)

            """
            rho_map_tng = np.zeros(nbins)
            n_map_tng_avg = 0
            tng_idcs = np.argwhere((np.log10(tng_halos['GroupMassType'][sel,dm_type])+10>=h_mass_range[0]) & (np.log10(tng_halos['GroupMassType'][sel,dm_type])+10<h_mass_range[1]))
            for idx in tng_idcs:
                idx = idx[0]
                if idx in gal_map_dict:
                    m_ids = [map_dict[mpid] for mpid in gal_map_dict[idx].map_pids] 
                    r = np.linalg.norm(maps_tup[1][m_ids,1:4] - tng_halos['GroupCM'][idx]/1e3,axis=1)
                    m = maps_tup[1][m_ids,0]*Mp.value
                    hist_counts = np.histogram(r[m>map_thresh], bins=rhist_edges,density=False, weights=m[m>map_thresh])[0]
                    if(np.any(np.isnan(hist_counts))):
                        continue
                    else:
                        rho_map_tng += hist_counts
                        n_map_tng_avg += np.sum(hist_counts)
            """
            
            # Get normalizations
            rho_tng = rho_tng / (n_tng_avg[0]) * rhist# * np.diff(rhist_edges)) * rhist**1  
            rho_map = rho_map / (n_map_avg[0]) * rhist # * np.diff(rhist_edges)) * rhist**1
            n_tng = n_tng / (n_tng_avg[1])  * rhist # * np.diff(rhist_edges)) * rhist**1
            n_map = n_map / (n_map_avg[1]) * rhist # * np.diff(rhist_edges)) * rhist**1 
            #rho_map_tng = rho_map_tng / (n_map_tng_avg * np.diff(rhist_edges)) * rhist**1
            
            # Make plots
            #ax[row,col].plot(rhist[rho_tng>0], rho_tng[rho_tng>0], '.-', c='red', lw=4,  ms=10, label=r'$\left\langle \frac{dM_*}{d\text{ln}(r)} \right\rangle$ MAP')
            ax[row,col].plot(rhist[n_tng>0], n_tng[n_tng>0], '.--', c='red', lw=4, ms=10, label=r'$\left\langle \frac{dN}{d\text{ln}(r)} \right\rangle$ TNG')
            
            #ax[row,col].plot(rhist[rho_map>0], rho_map[rho_map>0],'.-', c='blue', lw=4, ms=10, label=r'$\left\langle \frac{d\mathfrak{M}}{d\text{ln}(r)} \right\rangle$ MAP')
            ax[row,col].plot(rhist[n_map>0], n_map[n_map>0],'.--', c='blue', lw=4, ms=10,  label=r'$\left\langle \frac{dN}{d\text{ln}(r)} \right\rangle$ MAP')
            
            #ax[row,col].plot(rhist[rho_map_tng>0], rho_map_tng[rho_map_tng>0],'.-', c='tab:blue', label=r'MAP: $\langle n_{MAP} \rangle$')
            ax[row,col].grid()
            ax[row,col].set(xscale='log', yscale='linear')
            ylo,yhi = ax[row,col].get_ylim()
            xlo,xhi = ax[row,col].get_xlim()
            ax[row,col].text((xhi/xlo)**0.15*xlo, (yhi-ylo)*0.58+ylo, r'$\log_{{10}}(M_h/M_{{\odot}})=[{:.2f}-{:.2f}]$'.format(h_mass_range[0], h_mass_range[1]))
            if(row==nrow-1):
                ax[row,col].set_xlabel(xlabel=r'$r\;\;[h^{-1} Mpc]$')
            #if(col==0):
                #ax[row,col].set_ylabel(ylabel=r'$[h Mpc^{-1}]$')
            #if(col!=0):
            #    ax[row,col].set_yticklabels([])
            if(col==ncol-1 and row==0):
                ax[row,col].legend(loc = 'upper right')

    for row in range(nrow):
        for col in range(ncol):
            h_mass_range = halo_mass_bins[row*nrow+col]
            R200 = (G/H0.value**3/Omega_M)**(1/3.0) *  10**np.array([h_mass_range[0]/3,h_mass_range[1]/3])
            ylim = ax[row,col].get_ylim()
            ax[row,col].plot(np.full(2, R200[0]), ylim, ':', c='tab:red')
            ax[row,col].plot(np.full(2, R200[1]), ylim, 'r:')
            ax[row,col].set_ylim(ylim)

    #fig.legend((r'$\langle n_{halo}\rangle$', r'$\langle n_{MAP}\rangle$'))
    return fig