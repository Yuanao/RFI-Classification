# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 16:18:07 2020

@author: M.Yuan
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import math
#filename='1FP20180222_0-1GHz_Dec+41.1_drifting_0012_new.npz'
filename='FP20180222_0-1GHz_Dec+41.1_drifting_0012.npz'
name = str(filename).replace(".npz",".fits")
data0 = np.load(filename)
print(data0.files)
start=str(data0['start']).replace('b','').replace("'","")
rfi_ratio = len(data0['rfi_channel'])/3200.0
bandpass = data0['bandpass']
total_time=data0['time']
ra=str(data0['ra']).replace('b','').replace("'","")
dec =str(data0['dec']).replace('b','').replace("'","")
weight = data0['weight']
tot_num=len(weight)
#weight = np.sort(weight)[::-1]
weight = weight[weight>=0.01]
com_num = len(weight)

weight=[float('{0:.4f}'.format(100*i)) for i in weight]
#weight = data0['weight']
print(data0.files)
print(len(data0['rfi_channel']))
com = data0['component']
print(com.shape)
base = data0['basis']
com_scale = int(data0['component'].shape[1])
print(com_scale)
print(total_time)
#xt1='2017-12-19 11:33:59'
t_arr1=pd.date_range(start=start, periods=com_scale, freq='%s us'%(int(1e6*total_time/com_scale)))
t_arr2=pd.date_range(start=start, periods=com_scale//4, freq='%s us'%(int(1e6*total_time/com_scale*4)))

fig= plt.figure(figsize=(20, 12))
params={'axes.labelsize': '25',
                 'xtick.labelsize':'25',
                'ytick.labelsize':'25',
                'lines.linewidth':'0.5' ,
                'legend.fontsize': '25',}
pylab.rcParams.update(params)
font={'family':'serif',
      'style':'normal',
      'color':'black',
      'size':13}

ax11=fig.add_axes([0.05, 0.8, 0.8,0.15])
ax11.axis('off')
ax11.text(0,1.05,'FILE:',fontdict=font,fontsize=15)
ax11.text(0.05,1.05,name,fontsize=15,fontweight='light')
ax11.text(0.0,0.8, 'RA:',fontdict=font,fontsize=15)
ax11.text(0.05,0.8, ra,fontsize=15,fontweight='light')
ax11.text(0.0,0.55,'DEC:',fontdict=font,fontsize=15)
ax11.text(0.05,0.55, dec,fontsize=15,fontweight='light')
ax11.text(0.,0.3, 'Az:',fontdict=font,fontsize=15)
ax11.text(0.05,0.3, "$74.49^{\degree}$",fontsize=15,fontweight='light')
ax11.text(0.0,0.05,'UTC:',fontdict=font,fontsize=15)
ax11.text(0.05,0.05,'%s'%(start),fontsize=15,fontweight='light')

#ax11.text(0.0,0.55,'CENTERY FREQUENCY:',fontdict=font,fontsize=15)
#ax11.text(0.18,0.55, '500 MHz',fontsize=15,weight='light')
#ax11.text(0.0,0.3,'CHANNEL NUMBER:',fontdict=font,fontsize=15)
#ax11.text(0.16,0.3, '4096',fontsize=15,weight='light')
#ax11.text(0.0,0.05,'TOTAL TIME:',fontdict=font,fontsize=15)
ax11.text(0.40,0.05,'RFI CHANNELS:',fontdict=font,fontsize=15)
ax11.text(0.52,0.05,'%.2f%%'%(100*rfi_ratio),fontsize=15,weight='light')

ax11.text(0.40,1.05, 'TELESCOPE:',fontdict=font,fontsize=15)
ax11.text(0.50,1.05, 'FAST',fontsize=15,weight='light')
#ax11.text(0.40,0.8,'RFI TYPES:',fontdict=font,fontsize=15)
#ax11.text(0.485,0.8,'%s;'%('Impulsive'),fontsize=15,weight='light',color='red')
#ax11.text(0.56,0.8,'%s;'%('Periodic'),fontsize=15,weight='light',color='green')
#ax11.text(0.62,0.8,'%s'%('Non-stationary'),fontsize=15,weight='light',color='deepskyblue')
ax11.text(0.40,0.8,'TIME PER FILE:',fontdict=font,fontsize=15)
ax11.text(0.52,0.8,'%s s'%total_time,fontsize=15,weight='light')
ax11.text(0.40,0.55,'CENTERY FREQUENCY:',fontdict=font,fontsize=15)
ax11.text(0.58,0.55, '500 MHz',fontsize=15,weight='light')
ax11.text(0.40,0.3,'COMPONENT NUMBER:',fontdict=font,fontsize=15)
ax11.text(0.58,0.3, '%s'%tot_num,fontsize=15,weight='light')

ax22=fig.add_axes([0.76, 0.83, 0.2,0.15])
ax22.plot(np.linspace(100,899.75,3200),bandpass,'black',label='Bandpass (MHz)')
ax22.scatter((data0['rfi_channel']+400)/4.0,bandpass[data0['rfi_channel']],c='red',marker='x',s=15,label='RFI Channel')
ax22.tick_params(axis="x", labelsize=15)
ax22.legend(prop={'size': 10},framealpha=0.9)
ax22.set_yticks([])

#sporad_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise.npz')
#imp_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like.npz',encoding='latin1')
#period_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic.npz')
#sporad_data = np.load('1FP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise.npz')
#imp_data = np.load('1FP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like.npz',encoding='latin1')
#period_data = np.load('1FP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic.npz')
sporad_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise.npz')
imp_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like.npz',encoding='latin1')
period_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic.npz')
va_imp = imp_data['cindex_info']
va_imp_sum = sum(va_imp)
va_period = period_data['cindex_info']
va_period_sum = sum(va_period)
va_sporad = sporad_data['cindex_info']
va_sporad_sum = sum(va_sporad)
va = list(va_imp)+list(va_period)+list(va_sporad)
va = np.sort(np.array(va))[::-1]
va_sum = sum(va)
weights=[va_imp_sum/va_sum,va_period_sum/va_sum,va_sporad_sum/va_sum]
#ax33=fig.add_axes([0.57, 0.80, 0.21,0.15])
ax33=fig.add_axes([0.57, 0.82, 0.21,0.15])
ax33.tick_params(axis="y", labelsize=15)
#ax33.set_title('Variance-Ratio of RFI types', fontdict = font,fontsize=15)
ax33.text(-2.15,1.05,'Variance-Ratio of RFI types', fontdict = font,fontsize=15)
explode = [0,0,0]
colors=['red','green','deepskyblue']
labels=['Impulsive','Periodic','Non-Stationary']
ax33.pie(weights,colors=colors,explode=explode,autopct='%1.1f%%',shadow=False,startangle=250)
ax33.legend(labels,loc="lower center",ncol=3,prop={'size': 10},handlelength=0.8,
            handletextpad=0.1,columnspacing=0.1,borderaxespad=-1)

fig.subplots_adjust(hspace=0,wspace=0,left = 0.05,right = 0.96,bottom = 0.05,top = 0.8)
left, width = 0.05, 0.42
bottom, height = 0.12, .68
bottom_m = left_m = left+width+0.0
bottom_r = left_r = left+width+0.07
rect_left = [left, bottom, width, height]
rect_mid = [left_m, bottom, 0.07, height]
rect_right = [left_r,bottom,0.42,height]
ax=plt.axes(rect_left)
ax2=plt.axes(rect_right)
ax1=plt.axes(rect_mid)
ax.set_xlabel('Time (UTC)',fontsize=25)
ax.set_ylabel('RFI Weights',fontsize=25)
ax.set_yticks([])
ax1.set_xlabel('Ratio (e^)',fontsize=25)
ax1.set_yticks([])
#ax1.set_xticks([])
colors = ['#1a55FF','#ff7f0e','#2ca02c','#d62728','#9467bd',
          '#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']

weight_y=np.arange(com_num)[::-1]
ax1.plot(np.log(weight),weight_y,'-.',linewidth=1,color='black')
for i in range(com_num):
    ax1.scatter(np.log(weight[i]),weight_y[i],s=50,c=colors[i%10])
for sca_x, sca_y in zip(np.log(weight),weight_y):
    if sca_x>=max(np.log(weight))*0.7:
       print(sca_x)
       sca_x_=sca_x-0.3*max(np.log(weight))
    elif sca_x<=0.3*max(np.log(weight)):
       sca_x_=sca_x+0.3*max(np.log(weight))
    else:
       sca_x_=sca_x 
    ax1.annotate('%.2f%%'%(math.e**sca_x),xy=(sca_x_, sca_y),xytext=(0,-5),
                 textcoords='offset points',ha='center',va='top',fontsize=15) 


print(len(t_arr2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=20)
for i in range(com_num):
    comi = com[i].reshape(4096,len(com[i])//4096).sum(axis=1)
    x1 = (comi-min(comi))/(max(comi)-min(comi))
    ax.plot(t_arr2,x1+com_num-i,color=colors[i%10])
    #ax.text(t_arr2[0]+3700,0.5+com_num-i,'%s%%'%weight[i],fontsize=15)
#ax.set_xlim(-100,4096*4+1000)
#ax2 = fig.add_subplot(1, 2, 2)
ax2.set_yticks([])
ax2.set_xticks([0,800,1600,2400,3200])
ax2.set_xticklabels(['100','300','500','700','900'])
ax2.set_xlabel('Frequency (MHz)')
ax2.set_ylabel('RFI Bases')
ax2.yaxis.set_label_position("right")
for i in range(com_num):
    x2 = (base[i]-min(base[i]))/(max(base[i])-min(base[i]))
    index = np.arange(len(x2))[(np.array(x2)>=3*np.std(x2)+np.mean(x2))|(np.array(x2)<=-3*np.std(x2)+np.mean(x2))]
    ax2.plot(x2+com_num-i,linewidth=1.5,color=colors[i%10])
    ax2.scatter(index,x2[index]+com_num-i,s=20,color=colors[i%10],marker='x')
plt.savefig('6FP20180222_0-1GHz_Dec+41.1_drifting_0012.png',dpi=100)
plt.show()
