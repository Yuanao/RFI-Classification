import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.fftpack import fft
from sklearn.decomposition import PCA
from sklearn import preprocessing
from PyAstronomy import pyasl
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import sys
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy import units as u
import heapq
import pandas as pd
import matplotlib.pyplot as plt
import astropy.io.fits as pyfits
from decimal import Decimal
import ephem
import matplotlib.pylab as pylab
import itertools

name = sys.argv[1]

secperday = 3600 * 24
cindex_info_1=[]
cindex_1=[]
rfi_component_1=[]
rfi_basis_1=[]
f_info_1=[]
t_info_1=[]
t_info_1_=[]

cindex_info_2=[]
cindex_2=[]
rfi_component_2=[]
rfi_basis_2=[]
f_info_2=[]
t_info_2=[]

cindex_info_3=[]
cindex_3=[]
rfi_component_3=[]
rfi_basis_3=[]
f_info_3=[]
t_info_3=[]

def read_fits(name):
    filename=name.replace("\n","")
    if 1==1:
        hdulist = pyfits.open(filename)
        hdu0 = hdulist[0]
        hdu1 = hdulist[1]
        data1 = hdu1.data['data']
        tsamp = hdu1.header['TBIN']
        ra = hdu0.header['RA']
        dec = hdu0.header['DEC']
        print(ra,dec)
        a,b,c,d,e = data1.shape
        if c > 1:
                 data = data1[:,:,1,:,:].squeeze().reshape((-1,d))
        else:

                 data = data1.squeeze().reshape((-1,d))
        l, m = data.shape
        #data = data.reshape(l//64,64, d).sum(axis=1)
        data = data[:,400:3600]
#        print(data.shape)
        subintoffset = hdu1.header['NSUBOFFS']
        samppersubint = int(hdu1.header['NSBLK'])
        tstart = "%.18f" % (Decimal(hdu0.header['STT_IMJD']) + Decimal(hdu0.header['STT_SMJD'] + tsamp * samppersubint * subintoffset )/secperday )
        jd=float(tstart)+2400000.5
        date=ephem.julian_date('1899/12/31 12:00:00')
        djd=jd-date
        str1=ephem.Date(djd).datetime()
        str2=str1.strftime('%Y-%m-%d %H:%M:%S')
        #str2=str1.strftime('%H:%M:%S')
        t_total = tsamp*l
        '''
        ###get ha###
        obs = ephem.Observer()
        obs.date = str2
        obs.lon = longit
        obs.lat = latit
        obs.elevation = elev
        sun = ephem.Sun()
        sun.compute(obs)
        #jd = ephem.julian_date(dt)
        t = (jd - 2451545.0) / 36525
        theta = 280.46061837 + 360.98564736629 * (jd - 2451545) \
            + .000387933 * t**2 - t**3 / 38710000

        # hour angle (AA ch13)
        ha = (theta + longit - ra * 180 / ephem.pi) % 360
        '''
    return (data,str2,t_total,ra,dec,jd)

def detect_peaks(x, mph,n_peaks):
    x = np.atleast_1d(x).astype('float64')
    peaks_index=list(map(list(x).index, heapq.nlargest(n_peaks, x)))
    if len(peaks_index)<=0:
        print('no peaks found')
        return list(1)
    if min(peaks_index)<=mph or max(peaks_index)>=len(x)-50:
       peaks_index.remove(min(peaks_index))
       return peaks_index
    else:
        peaks_index=list(np.delete(peaks_index,np.where(np.array(peaks_index)<mph)[0]))
        mph_arr=np.arange(-mph,mph+1)
        mph_arr=np.delete(mph_arr,mph)
        no_peaks=[]
        for i in peaks_index:
            dx=[x[i]-x[i+j] for j in mph_arr]
            if (np.array(dx)>0).all()==False:
               no_peaks.append(i)
            else:
               pass
        for i in no_peaks:
            peaks_index.remove(i)
        return peaks_index

def judge_continue(array):
    diff = array[1:]-array[:-1]
    #right_edge = np.where(diff!=1)[0]
    right_edge = np.where(diff>50)[0]
    array_split = np.split(np.array(array),np.array(right_edge)+1)
    #out_put = [int(np.median(array_split[i])) for i in range(len(array_split))]
    out_put = [list(array_split[i]) for i in range(len(array_split))]
    return out_put
def judge_in_out(array0,array1):
    array0 = list(array0)
    array1 = list(array1)
    array0 = list(itertools.chain.from_iterable(array0))
    #array1 = list(itertools.chain.from_iterable(array1))
    array0_ = np.array([np.arange(i-20,i+20) for i in array0]).reshape(-1)
    list0 = list(array0_)
    #print('array0',array0_)
    #print('array1',array1)
    num = len(set(list0) & set(array1))
    return num
### RFI classify and plot #####
def classify_rfi(data):
    ###detree1###
    data = (data-max(data))/(max(data)-min(data))
    std = np.std(data)
    mean = np.mean(data)
    median = np.median(data)
    data_fft = abs(fft(data))[1:len(data)//2]
    detree1=data[(data>=9*std+median)].size
    detree11 = max(data)/np.median(data)
    if detree1>=1:
       rfi_type='Impulsive'
       return rfi_type,data_fft
    else:
       ###detree2###
       peaks_index=list(map(list(data_fft).index, heapq.nlargest(15, data_fft)))
       detree2 = max(peaks_index)
       if detree2 <=30:
          rfi_type='Sporadic'
          return rfi_type,data_fft
       else:
             ###detree3###
             peak_x = detect_peaks(data_fft,mph=50,n_peaks=15)
             mean50 = np.mean(data_fft[:50])
             mean_peaks = np.mean(data_fft[peak_x])
             peak_x_mean = np.mean(peak_x)
             detree3_1 = mean_peaks/np.mean(data_fft[50:])
             detree3_2 = mean_peaks/mean50
             detree3_3 = max(data_fft[50:])/mean50
             detree3 = detree3_1+detree3_2+detree3_3
             if detree3 <= 10:
                rfi_type = 'Sporadic'
                return rfi_type,data_fft
             else:
                if peak_x_mean<30:
                   rfi_type = 'Sporadic'
                   #return rfi_type,data_fft
                else:
                   std1 = np.std(data_fft[30:])
                   if std1*detree3<10:
                      rfi_type = 'Sporadic'
                   else:
                      rfi_type = 'Periodic'
                return rfi_type,data_fft

####plot rfi ####
def plot_rfi(t_data,t_info,f_info,cindex_info,rfi_type,t_text,f_text):
    t_data = np.array(t_data)
    rfi_num = len(cindex_info)
    if rfi_num==0:
        pass
    elif rfi_num==1:
        fig, axs = plt.subplots(1,2, sharex=False)
        fig.suptitle('%s'%rfi_type,y=0.95,fontsize=20)
        fig.subplots_adjust(hspace=0,wspace=0)
        fig.autofmt_xdate()
        axs[0].plot(t_arr,t_data[0],color='black')
        #axs[0].plot(t_arr,np.zeros(len(t_data[0]))+np.mean(t_data[0])+5*np.std(t_data[0]))
        axs[0].set_yticks([np.max(t_data[0])/2])
        axs[0].set_yticklabels(['%.2e'%cindex_info[0]],rotation=90)
        axs[1].spines['right'].set_visible(False)
        axs[1].spines['top'].set_visible(False)
        axs[1].set_xticks([])
        axs[1].set_yticks([])
        t_note = str("%s%s"%(t_text,t_info)).replace("\n"," ").replace("[","").replace("]","")
        f_note = str("%s%s"%(f_text,f_info)).replace("\n"," ").replace("[","").replace("]","")
        axs[1].text(0.005,0.95,t_note,ha='left',va='top',fontsize=5,wrap=True,transform=axs[1].transAxes)
        axs[1].text(0.005,0.7,f_note,fontsize=5,ha='left',va='top',wrap=True,transform=axs[1].transAxes)
        plt.savefig('%s_%s.png'%(basename,rfi_type),dpi=800)
        plt.close('all')
    elif rfi_type>1:
        fig, axs = plt.subplots(rfi_num,2, sharex=False)
        fig.subplots_adjust(hspace=0,wspace=0)
        fig.autofmt_xdate()
        for i in range(rfi_num):
    # Plot each graph, and manually set the y tick values
            t_note = str("%s%s"%(t_text,t_info[i])).replace("\n"," ").replace("[","").replace("]","")
            f_note = str("%s%s"%(f_text,f_info[i])).replace("\n"," ").replace("[","").replace("]","")
            axs[i,0].plot(t_arr,t_data[i],color='black')
    #        axs[i,0].plot(t_arr,np.zeros(len(t_data[i]))+np.mean(t_data[i])+5*np.std(t_data[i]))
            axs[i,0].set_yticks([np.max(t_data[i])/2])
            axs[i,0].set_yticklabels(['%.2e'%cindex_info[i]],rotation=-60)
            axs[i,1].set_xticks([])
            axs[i,1].set_yticks([])
            axs[i,1].spines['right'].set_visible(False)
            axs[i,1].spines['top'].set_visible(False)
            axs[i,1].text(0.005,0.95,t_note,fontsize=5,wrap=True,ha='left',va="top",transform=axs[i,1].transAxes)
            axs[i,1].text(0.005,0.2,f_note,fontsize=5,wrap=True,ha='left',va="top",transform=axs[i,1].transAxes)
    #        plt.tight_layout()
        plt.savefig('%s_%s.png'%(basename,rfi_type),dpi=800)
        plt.close('all')

name = name.replace('\n','')
basename = name.replace('.fits','')
data_raw,str2,t_total,ra,dec,jd = read_fits(name)
l,m=data_raw.shape
data = data_raw.reshape(l//16,16, 3200).sum(axis=1)
#np.savez('%s_data'%basename,data,data)


bandpass = data.sum(axis=0)
RA = ra
Dec =dec
fast = SkyCoord(RA,Dec,unit= (u.hourangle,u.deg)) 
ra = fast.ra.degree
dec = fast.dec.degree
obstime = Time(str2)
#da = RA(ra,unit=u.degree)
#dec = Dec(dec,unit=u.degree)
#fast_location = EarthLocation(lat = 25.652939, lon = 106.856594, height = 1096)
#fast = SkyCoord(RA,Dec,unit=(u.hourangle, u.deg))
#fast_altaz = fast.transform_to(AltAz(obstime=obstime,location=fast_location))
alt, az, ha = pyasl.eq2hor(jd,ra,dec,lat = 25.652939, lon = 106.856594, alt = 1096)
alt=alt[0]
az=az[0]
altout = [alt,az]
t_re,f_re = data.shape
f_max=t_re/t_total
f_min=1/t_total
f_p = np.linspace(f_min,f_max,t_re//2)[30:]
pca = PCA(0.95)
#pca = PCA(200)
pca.fit(data)
#base = pca.transform(data.T)[1:,:]
base = pca.transform(data)
arr1 = np.array(pca.components_)
print('component',arr1.shape)
arr2 = np.array(base)
###########
arr11 = arr1.copy()
arr11[abs(arr11)<=0.3]=0
sum_rise=arr11.sum(axis=0)
sum_abs=abs(arr11).sum(axis=0)
index_rise=np.where((sum_abs>=0.9) & (sum_rise>0))[0]
index_down=np.where((sum_abs>=0.9)& (sum_rise<0))[0]
print(index_rise,index_down)
for i in index_rise:
     i=int(i)
     com_index1=np.where(arr11[:,i]<max(arr11[:,i]))[0]
     arr1[com_index1,i]=0
for i in index_down:
     i=int(i)
     com_index3=np.where(arr11[:,i]>min(arr11[:,i]))[0]
     arr1[com_index3,i]=0
#
#n_ = int(arr1.shape[0])
#for i in range(n_):
#    max_ch = int(list(abs(arr1[i,:])).index(max(abs(arr1[i,:]))))
#    max_ch_va = abs(arr1[i,max_ch])
#    if max_ch_va==max(abs(arr1[:,max_ch])):
#       print('yes',max_ch,arr1[i,max_ch])
#       arr1[i+1:n_,max_ch]=0
#       arr1[0:i-1,max_ch]=0
#       print((max_ch,arr1[n_-1,max_ch]))
#    else:
#       print('no')
base = np.dot(data,arr1.T)
print(base.shape)
weight_arr = [np.std(base[:,i])**2 for i in range(pca.n_components_)]
weight_arr_ratio = np.array(weight_arr)/sum(weight_arr)
weight_sort = np.argsort(-weight_arr_ratio)
arr1=np.array([arr1[i] for i in weight_sort])
weight=weight_arr_ratio[weight_sort]
base=np.array([base[:,i] for i in weight_sort])

###########
#print(arr1.shape,arr2.shape)
#print('basis',arr2.shape)
arr3 = str(str2)
#print('time',str2)
#np.savez("%s"%basename,component=pca.components_,basis=base,start=str2,time=t_total)

fig= plt.figure()
params={
'axes.labelsize': '10',
'xtick.labelsize':'10',
'ytick.labelsize':'10',
'lines.linewidth':'0.3' ,
'legend.fontsize': '10',
}
pylab.rcParams.update(params)

t_arr=pd.date_range(start=str2, periods=t_re, freq='%s us'%(int(1e6*t_total/t_re)))

plt.subplots_adjust(wspace=0.002)
ax = fig.add_subplot(1, 2, 2)
fig.autofmt_xdate()
ax.set_xlabel('Time')
ax.set_ylabel('Components')
ax3 = fig.add_subplot(1, 2, 1)
ax3.set_ylabel('Basis')
ax3.set_xlabel('Frequency (MHz)')
#ax3.set_xticks(np.linspace(0,4096,6))
ax3.set_yticks([])
#ax3.set_xticklabels([0,200,400,600,800,1000])
ax3.yaxis.set_label_position("left")
arr2=np.array(arr2)
index=[]
com_num = len(weight[weight>0.001])
base=base[:com_num,:]
arr1=arr1[:com_num,:]
print('com_num',com_num)
for j in range(com_num):
     y_arr1 = (arr1[j]-min(arr1[j]))/(max(arr1[j])-min(arr1[j]))+(20-j)
     ax3.plot(y_arr1,linewidth=0.3)
     index0=list(np.array(np.where(abs(arr1[j])>3*np.std(arr1[j]))).reshape(-1))
     ax3.scatter(index0,y_arr1[index0],s=0.1)
     index=index+index0
c=sorted(list(set(np.array(index))))
impulsive_count=-1
max_ch=[]
impulse_index=[]
pulse_ch1=[]
for i in range(com_num):
    ratio = pca.explained_variance_ratio_[i]
    #a=pca.components_[i]
    #a= base[i]
    #a = base[:,i]
    #print('component:%s'%i)
   # f_rfi = np.where(np.array(arr2)[:,i]>=3*np.std(np.array(arr2)[:,i]))[0].tolist()
    arr22 = np.array(np.array(arr1)[i])
    f_rfi = int(list(arr22).index(max(arr22))+400)
    while f_rfi in f_info_1:
#          print('impulsive yes',i)
          arr22[f_rfi-400]=0
          f_rfi = list(abs(arr22)).index(max(abs(arr22)))+400
#          print(f_rfi)
    while f_rfi in f_info_2:
#          print('periodic yes',i)
          arr22[f_rfi-400]=0
          f_rfi = list(abs(arr22)).index(max(abs(arr22)))+400
#          print(f_rfi)
    while f_rfi in f_info_3:
#          print('non-stationary yes',i)
          arr22[f_rfi-400]=0
          f_rfi = list(abs(arr22)).index(max(abs(arr22)))+400
#          print(f_rfi)
    a = np.dot(data,arr22)
    a = list(a)
    a = list((a-min(a))/(max(a)-min(a)))
    #####classify rfi #####
    rfi_type,data_fft = classify_rfi(a)
    #print(rfi_type)
    #a = list(a)
    #a = list((a-min(a))/(max(a)-min(a)))
    arr2 = list(arr2)
    if rfi_type=='Impulsive':
       impulsive_count+=1
       #print(np.where(abs(np.array(arr22))>np.mean(arr22)+7*np.std(np.array(arr22))))
       pulse_ch = list(np.where(abs(np.array(arr22))>np.mean(arr22)+7*np.std(np.array(arr22)))[0])
       #print(pulse_ch)
       f_rfi = int(list(arr22).index(max(arr22))+400)
       x_arr = np.arange(len(a))
       imp_index = x_arr[(a>=(5*np.std(a)+np.mean(a)))]
       imp_index_arr_output = judge_continue(imp_index)
       imp_index_p =  [np.where(np.array(a)==max(np.array(a)[q]))[0][0] for q in imp_index_arr_output]
       t_rfi0=t_arr[imp_index_p]
       ###calculate stn of pulses###
       acopy = a[:]
       [acopy.remove(a[j]) for j in list(itertools.chain.from_iterable(imp_index_arr_output))]
       amean = np.mean(acopy)
       stn_pulse = [float(np.round(a[p]/amean,2)) for p in imp_index_p]
       #print(stn_pulse)
       t_rfi1 = list(t_rfi0.strftime('%Y-%m-%d %H:%M:%S'))
       t_rfi2 = sorted(set(t_rfi1),key=t_rfi1.index)
       #print('impulsive:%s , %s, %s'%(impulsive_count,t_rfi2,f_rfi))
       #print(stn_pulse)
       imp_index_p = list(imp_index_p)
       t_rfi2 = list(t_rfi2)
       stn_pulse = list(stn_pulse)
       t_rfi = list([imp_index_p,t_rfi2,stn_pulse])
       
       if impulsive_count<1:
          pulse_ch1.append(pulse_ch)
          #print(pulse_ch1)
          cindex_info_1.append(weight[i])
          cindex_1.append(i)
          rfi_component_1.append(a)
          rfi_basis_1.append(list(arr22))
          f_info_1.append(f_rfi)
          t_info_1_.append(t_rfi2)
          t_info_1.append(t_rfi)
          impulse_index.append(imp_index_p)
       else:
          #same_pulse_num = judge_in_out(impulse_index,imp_index_p)
          same_pulse_num=[1.0*len(set(imp_index_p) & set(impulse_index[i]))/len(imp_index_p) for i in range(len(impulse_index))]  
          #print(imp_index_p,impulse_index,same_pulse_num) 
          if max(same_pulse_num)<0.7:
          #if t_rfi2 not in t_info_1_:
             pulse_ch1.append(pulse_ch)
             #print('no repeating',pulse_ch1)
             cindex_info_1.append(weight[i])
             cindex_1.append(i)
             rfi_component_1.append(a)
             rfi_basis_1.append(list(arr22))
             f_info_1.append(f_rfi)
             t_info_1_.append(t_rfi2)
             t_info_1.append(t_rfi)
             impulse_index.append(imp_index_p)
          else:
               # print('component %s, same with component %s'%(impulsive_count+1,impulsive_count))
                #wide_com_num=t_info_1_.index(t_rfi2)
                #print(t_rfi2)
                wide_com_num=same_pulse_num.index(max(same_pulse_num))
                cindex_info_1[wide_com_num]=cindex_info_1[wide_com_num]+weight[i]
                impulse_index[i] = list(set(imp_index_p).union(set(impulse_index[i])))
                f_info_int = np.array([int(ele) for ele in f_info_1 if ele!='whole band'])
                #if min(abs(f_info_int-f_rfi))>20 and cindex_info_1[wide_com_num] >=0.05:
                if min(abs(f_info_int-f_rfi))>20:
                   #print('repeating whole band')
                   f_info_1[wide_com_num]='whole band'
                else:
                   #print('repeating narrow band')
                   pulse_ch1.append(pulse_ch)
                   #print('repeating,narrow',pulse_ch)
    if rfi_type=='Periodic':
       f_rfi = list(abs(arr22)).index(max(abs(arr22)))+400
       #data_fft0=data_raw[:,f_rfi]
       data_fft1=a
       #data_fft1=data_fft0.reshape(len(data_fft0)//16,16).sum(axis=1)
       ef_min = 2/t_total
       ef_max = len(data_fft1)/t_total/2
       ef_x = np.linspace(ef_min,ef_max,len(data_fft1)//2)
       ef_x = np.round(ef_x,2)
       data_fft2=list(abs(fft(data_fft1))[40:len(data_fft1)//2])
       ef_x = ef_x[40:]
       t_rfi = ef_x[data_fft2==max(data_fft2)][0]
       cindex_info_2.append(weight[i])
       cindex_2.append(i)
       rfi_component_2.append(a)
       rfi_basis_2.append(list(arr22))
       f_info_2.append(f_rfi)
       t_info_2.append(t_rfi)
    if rfi_type=='Sporadic':
       f_rfi = list(abs(arr22)).index(max(abs(arr22)))+400
       cindex_info_3.append(weight[i])
       cindex_3.append(i)
       rfi_component_3.append(a)
       rfi_basis_3.append(list(arr22))
       f_info_3.append(f_rfi)
       t_info_3.append('None')
    a_norm = (a-min(a))/(max(a-min(a)))
    a_abs = abs(np.array(a))
    a_abs = a_abs.tolist()
    ax.plot(t_arr,a_norm+(10-i),linewidth=0.3)
    #result = map(a_abs.index, heapq.nlargest(3, a_abs))
#    a = np.array(a)
#    ax3.scatter(index0,y_arr2[index0],s=0.1)
##   index=index+index0
#plt.show()
plt.savefig('%s.png'%basename,dpi=800)
#c=sorted(list(set(np.array(index))))
##arr = np.zeros(4096)
##arr[index]=1
##np.savetxt('%s.txt'%basename,arr,fmt='%s')
'''
if len(f_info_1)<1:
   impulsive=None
elif len(f_info_1)>=1 and type(f_info_1[0])=='int':
   impulsive=[data[i-400,:] for i in f_info_1]
else:
   impulsive=[data[i-400,:] for i in f_info_1[1:]]
'''
print('Impulse-like:%s'%(len(cindex_info_1)))
print('Periodic:%s'%(len(cindex_info_2)))
print('Colored-noise:%s'%(len(cindex_info_3)))
cindex_info = cindex_info_1+cindex_info_2+cindex_info_3
#print(cindex_info)
resort_index = np.argsort(cindex_info)[::-1]
com_list=[]
base_list=[]
for com_list0 in [rfi_component_1,rfi_component_2,rfi_component_3]:
    if len(com_list0)==0:
       pass
    else:
       com_list+=com_list0
       #com_list.append(com_list0)
component = np.array(com_list)
#print(component)
for base_list0 in [rfi_basis_1,rfi_basis_2,rfi_basis_3]:
    if len(base_list0)==0:
       pass
    else:
       #print(len(base_list0))
       base_list+=base_list0
base = np.array(base_list)
#base = np.array(base)
#print('base.shape',np.array(base).shape)
'''
if len(cindex_info_1)==0:
   base = np.vstack((np.array(rfi_basis_2),np.array(rfi_basis_3)))
   component = np.vstack((np.array(rfi_component_2),np.array(rfi_component_3)))
else:
   base = np.vstack((np.array(rfi_basis_1),np.array(rfi_basis_2),np.array(rfi_basis_3)))
   component = np.vstack((np.array(rfi_component_1),np.array(rfi_component_2),np.array(rfi_component_3)))
'''
#print(resort_index)
base = [base[i] for i in resort_index]
components = [component[i] for i in  resort_index]
weights=[cindex_info[i] for i in  resort_index]
np.savez("%s"%basename,bandpass=bandpass,rfi_channel=c,component=components,basis=base,weight=weights,start=str2,time=t_total,ra=RA, dec=Dec)
#np.savez("%s"%basename,bandpass=bandpass,rfi_channel=c,component=base,basis=arr1,weight=weight,start=str2,time=t_total,ra=RA, dec=Dec)
plt.close('all')
#### save rfi info ####
np.savez("%s_Impulse-like"%basename,direction=altout,cindex_info=cindex_info_1,cindex=cindex_1,rfi_component=rfi_component_1,rfi_basis=rfi_basis_1,f_info=f_info_1,t_info=t_info_1)
np.savez("%s_Periodic"%basename,direction=altout,cindex_info=cindex_info_2,cindex=cindex_2,rfi_component=rfi_component_2,rfi_basis=rfi_basis_2,f_info=f_info_2,t_info=t_info_2)
np.savez("%s_Colored-noise"%basename,direction=altout,cindex_info=cindex_info_3,cindex=cindex_3,rfi_component=rfi_component_3,rfi_basis=rfi_basis_3,f_info=f_info_3,t_info=t_info_3)
#### plot classified RFI ####
###plot impulse-like rfi
plot_rfi(rfi_component_1,t_info_1,f_info_1,cindex_info_1,'Impulse-like','The occurence time of impule_like RFI:','The occurence frequency band of impule_like RFI:')
plot_rfi(rfi_component_2,t_info_2,f_info_2,cindex_info_2,'Periodic','The base frequency of periodic RFI:','The occurence frequency band of periodic RFI:')
plot_rfi(rfi_component_3,t_info_3,f_info_3,cindex_info_3,'Colored-noise','','The occurence frequency band of colored-noise:')

if len(f_info_1) == 0:
   pass
else:
   rfi_num = len(f_info_1)
   total_in = data.sum(axis=1)
   fig, axs = plt.subplots(rfi_num+1,1, sharex=False)
   fig.subplots_adjust(hspace=0,wspace=0)
   o=-1
   for f in f_info_1:
       o+=1
       if f!='whole band':
          ch = int(f)-400
          test = data[:,ch]
          axs[o].plot(test)
          axs[o].text(100,0.8*max(test),'%s'%(ch+400))
       else:
          axs[o].plot(total_in)  
          axs[o].text(100,0.99*max(total_in),'whole band')
   axs[rfi_num].plot(total_in)
   axs[rfi_num].text(100,0.99*max(total_in),'Total intensity')
   plt.savefig('%s_pulse.png'%(basename),dpi=200)
plt.close('all')
