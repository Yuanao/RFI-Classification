# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 18:42:21 2020

@author: M.Yuan
"""

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pylab
import warnings 
warnings.filterwarnings("ignore", category=UserWarning)
import matplotlib.text as mtext
import pandas as pd
from scipy.fftpack import fft
from adjustText import adjust_text

params={
'axes.labelsize': '10',
'xtick.labelsize':'25',
'ytick.labelsize':'25',
'lines.linewidth':'0.5' ,
'legend.fontsize': '20',
}
pylab.rcParams.update(params)

font={'family':'serif',
      'style':'normal',
      'color':'black',
      'size':13}

class WrapText(mtext.Text):
    def __init__(self,
                 x=0, y=0, text='',
                 width=0,
                 **kwargs):
        mtext.Text.__init__(self,
                 x=x, y=y, text=text,
                 wrap=True,
                 **kwargs)
        self.width = width  # in screen pixels. You could do scaling first

    def _get_wrap_line_width(self):
        return self.width
    
#data_a = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012.npz')
#imp_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like.npz',encoding='latin1',allow_pickle=True)
#period_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic.npz')
#sporad_data = np.load('newFP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise.npz')
data_a = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012.npz')
imp_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like.npz',encoding='latin1',allow_pickle=True)
period_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic.npz')
sporad_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise.npz')
#data_a = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_new.npz')
#imp_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Impulse-like_new.npz',encoding='latin1',allow_pickle=True)
#period_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Periodic_new.npz')
#sporad_data = np.load('FP20180222_0-1GHz_Dec+41.1_drifting_0012_Colored-noise_new.npz')
period_com = period_data['rfi_component']
period_base = period_data['rfi_basis']
sporad_com = sporad_data['rfi_component']
sporad_base = sporad_data['rfi_basis']
base=data_a['basis']
base_scale = int(data_a['component'].shape[1])
print(base_scale)
start=str(data_a['start']).replace('[','').replace('b','')
t_total = data_a['time']

print(data_a.files)
t_arr1=pd.date_range(start=start, periods=base_scale, freq='%s us'%(int(1e6*t_total/base_scale)))
t_arr2=pd.date_range(start=start, periods=base_scale//4, freq='%s us'%(int(1e6*t_total/base_scale//4)))
date_list = [x.strftime('%H:%M:%S') for x in t_arr1]
fft_x = np.linspace(2/t_total,base_scale/t_total/2,base_scale//2)

va_imp = imp_data['cindex_info']
va_imp_sum = sum(va_imp)
va_period = period_data['cindex_info']
va_period_sum = sum(va_period)
va_sporad = sporad_data['cindex_info']
va_sporad_sum = sum(va_sporad)
va = list(va_imp)+list(va_period)+list(va_sporad)
va = np.sort(np.array(va))[::-1]
va_sum = sum(va)
def plot_classified_imp(data):
    f_arr=(np.arange(3200)+400)/4
    imp_data = data
    imp_com = imp_data['rfi_component']
    imp_base = imp_data['rfi_basis']
    imp_finfo = imp_data['f_info']
    print(imp_data['t_info'])
    imp_finfo=[str(imp_finfo[i]).replace('b','').replace("'","") for i in range(len(imp_finfo))]
    imp_num0 = imp_com.shape[0]
    weights=[va_imp_sum/va_sum,va_period_sum/va_sum,va_sporad_sum/va_sum]
    if imp_num0 == 0:
        pass
    if imp_num0 >10:
        imp_num=10
    else: 
        imp_num=imp_num0
    fig0, ax0 = plt.subplots(imp_num, 2, sharex=False, sharey=True,figsize=(20, 12))
    fig0.subplots_adjust(hspace=0,wspace=0,left = 0.01,right = 0.99,bottom = 0.13,top = 0.8)
    ax1=fig0.add_axes([0.01, 0.85, 0.8,0.145])
    ax3=fig0.add_axes([0.28, 0.83, 0.34,0.16])
    ax2=fig0.add_axes([0.65, 0.83, 0.34,0.16])
    ax1.axis('off')
    text_type = 'RFI type:'
    text_type_ = 'Impulsive'
    text_com = 'Component Number: %s'%(imp_num)
    text_va = 'Sum of Variance-Ratio: %1.1f%%'%(weights[0]*100)
    ax1.text(0,0.75,text_type,fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0.1,0.75,text_type_,color='red',fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0,0.45,text_com,fontdict=font,fontsize=20)
    ax1.text(0,0.15,text_va,fontdict=font,fontsize=20)
    va_x = np.arange(len(va))+1
    ax2.plot(va_x,np.log(va),'.',linewidth=0.5,color='black')
    for ele in np.log(list(va_imp)):
        ax2.scatter(va_x[np.log(va)==ele],ele,color='red',s=50)
    ax2.tick_params(axis="x", labelsize=15)
    ax3.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax3.tick_params(axis="y", labelsize=15)
    ax2.text(len(va)/5,1.8*max(np.log(va)),'Variance-Ratio of RFI Weights',fontdict=font,fontsize=15,weight='light')
#    ax2.xaxis.set_ticks_position('top')
#    ax2.yaxis.set_ticks_position('right')
#    ax2.tick_params(axis="y",direction="in")
#    plt.rcParams['ytick.direction']= 'in'
#    ax2.xaxis.set_ticks_position('top')
    
#    explode = [0,0.2,0.2]
#    ax3.set_title('Variance-Ratio of RFI types', fontdict = font,fontsize=15) 
#    ax3.pie(weights,colors=colors,explode=explode,autopct='%1.1f%%',shadow=True,startangle=250)
    
    
    print('Impulsive:%s'%(imp_num))
    for i in range(imp_num):
        plot_data0 = (imp_com[i]-min(imp_com[i]))/(max(imp_com[i])-min(imp_com[i]))
        plot_data1 = (imp_base[i]-min(imp_base[i]))/(max(imp_base[i])-min(imp_base[i]))
        ax0[i,0].plot(t_arr1,plot_data0,'black')
        imp_info_index = np.array([int(i) for i in imp_data['t_info'][i][0]])        
        ax0[i,0].scatter(t_arr1[imp_info_index],plot_data0[imp_info_index],s=50,color='red')
        f_node_ = 'Frequency: '
        if imp_finfo[i]=='whole and':
           f_note = str("Frequency: whole-band") 
           ax0[i,1].text(70,0.85, f_note,fontsize=14)
           f_disy=np.linspace(100,900,3200) ############
           for pulse in imp_info_index: ############
               print(pulse)
               f_disx=[t_arr1[pulse] for i in range(3200)] ############
               ax3.scatter(f_disx,f_disy,marker='s',s=10,c='red') ############
               ax0[i,1].plot(f_arr,plot_data1,'black')
        else:
           for pulse in imp_info_index: ############
               ax3.scatter(t_arr1[pulse],int(imp_finfo[i])/4,marker='s',s=20,c='red') ############
           ax0[i,1].plot(f_arr,plot_data1,'black')
           
           ax0[i,1].scatter((int(imp_finfo[i]))/4,plot_data1[int(imp_finfo[i])-400],s=50,color='red')
           f_note = str("%s%s MHz"%(f_node_,int(imp_finfo[i])/4)).replace("\n"," ").replace("[","").replace("]","")
           if int(imp_finfo[i])-400<=len(plot_data1)/2:
              ax0[i,1].text((int(imp_finfo[i])+100)/4,0.85, f_note,fontsize=14,fontweight='light')
           if int(imp_finfo[i])-400>len(plot_data1)/2:
              ax0[i,1].text((int(imp_finfo[i])-100)/4,0.85, f_note,fontsize=14,fontweight='light')
        ax0[i,0].set_yticks([])
        ax0[i,1].set_yticks([])
        if i<imp_num-1:
           ax0[i,0].set_xticks([])
           ax0[i,1].set_xticks([])
        plt.setp(ax0[i,0].xaxis.get_majorticklabels(), rotation=20)
    ax0[i,1].set_xlabel('Frequency(MHz)',fontsize=25) 
    ax0[i,0].set_xlabel('Time (UTC)',fontsize=25) 
    plt.savefig('6impulsive.png',dpi=100) 
    
def plot_classified_period(data):
    f_arr=(np.arange(3200)+400)/4
    period_data = data
    period_com = period_data['rfi_component']
    period_base = period_data['rfi_basis']
    period_finfo = period_data['f_info']/4
    #period_finfo=[str(period_finfo[i]).replace('b','').replace("'","") for i in range(len(period_finfo))]
    period_num0 = period_com.shape[0]
    weights=[va_imp_sum/va_sum,va_period_sum/va_sum,va_sporad_sum/va_sum]
    if period_num0 == 0:
        pass
    if period_num0 >10:
        period_num=10 
    else:
        period_num=period_num0
    fig0, ax0 = plt.subplots(period_num, 3, sharex=False, sharey=True,figsize=(20, 12))
    fig0.subplots_adjust(hspace=0,wspace=0,left = 0.01,right = 0.99,bottom = 0.13,top = 0.8)
    ax1=fig0.add_axes([0.01, 0.85, 0.8,0.145])
    ax3=fig0.add_axes([0.28, 0.83, 0.34,0.16])
    ax2=fig0.add_axes([0.65, 0.83, 0.34,0.16])
    ax2.tick_params(axis="x", labelsize=15)
    ax3.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax3.tick_params(axis="y", labelsize=15)
    for period_finfo_ in period_finfo:
        f_disy = np.zeros(4096*4)+period_finfo_ ############
        ax3.plot(t_arr1,f_disy,c='deepskyblue')
    ax1.axis('off')
    text_type = 'RFI type:'
    text_type_ = 'Periodic'
    text_com = 'Component Number: %s'%(period_num0)
    text_va = 'Sum of Variance-Ratio: %1.1f%%'%(weights[1]*100)
    ax1.text(0,0.75,text_type,fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0.1,0.75,text_type_,color='deepskyblue',fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0,0.45,text_com,fontdict=font,fontsize=20)
    ax1.text(0,0.15,text_va,fontdict=font,fontsize=20)
    va_x = np.arange(len(va))+1
    ax2.plot(va_x,np.log(va),'.',linewidth=0.5,color='black')
    for ele in np.log(list(va_period)):
        ax2.scatter(va_x[np.log(va)==ele],ele,color='deepskyblue',s=50)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax2.text(len(va)/5,1.8*max(np.log(va)),'Variance-Ratio of RFI Weights',fontdict=font,fontsize=15,weight='light')  
    print('Periodic:%s'%(period_num))
    for i in range(period_num):
        plot_data0 = (period_com[i]-min(period_com[i]))/(max(period_com[i])-min(period_com[i]))
        ax0[i,0].plot(t_arr1,plot_data0,'black')
        f_node_ = 'Frequency: '
        plot_data1 = (period_base[i]-min(period_base[i]))/(max(period_base[i])-min(period_base[i]))
        ax0[i,1].plot(f_arr,plot_data1,'black',linewidth=0.5)
        ax0[i,1].scatter(period_finfo[i],plot_data1[int(period_finfo[i]*4)-400],s=50,c='deepskyblue')
        if period_finfo[i] >750:
           text_x =  period_finfo[i]-400
        elif period_finfo[i] <=200:
           text_x =  period_finfo[i]+200
        else:
           text_x =  period_finfo[i]+50 
        if plot_data1[int(period_finfo[i]*4)-400]>0:
            text_y = np.mean(plot_data1)+0.3
        if plot_data1[int(period_finfo[i]*4)-400]<=0:
            text_y = np.mean(plot_data1)-0.3    
        ax0[i,1].annotate("%s%s MHz"%(f_node_,period_finfo[i]),xy=(period_finfo[i],plot_data1[int(period_finfo[i]*4)-400]),
           xytext=(text_x,text_y),fontsize=15)
#        f_note = str("%s%s MHz"%(f_node_,period_finfo[i]))
#        if int(period_finfo[i])<=500:
#           ax0[i,1].text(int(period_finfo[i])+100,0.9, f_note,fontsize=14,fontweight='light')
#        if int(period_finfo[i])>500:
#              ax0[i,1].text(int(period_finfo[i])-100,0.9, f_note,fontsize=14,fontweight='light')
        fft_y0 = abs(fft(plot_data0))[0:base_scale//2]
        fft_y =fft_y0[30:]
        fft_x = np.linspace(1/52.466,1/0.00320,len(fft_y0))[30:]
        fft_y = (fft_y-min(fft_y))/(max(fft_y)-min(fft_y))
        max_y_index = np.where(fft_y==max(fft_y))[0]
        ax0[i,2].scatter(fft_x[max_y_index],fft_y[max_y_index],s=50,c='deepskyblue')
        if max_y_index>=len(fft_y)/2:
           ax0[i,2].annotate("Period: %.2fms"%(1000.0/fft_x[max_y_index]),xy=(fft_x[max_y_index],fft_y[max_y_index]),
           xytext=(fft_x[max_y_index]-100,fft_y[max_y_index]-0.33),fontsize=15)
        else:
           ax0[i,2].annotate("Period: %.2fms"%(1000.0/fft_x[max_y_index]),xy=(fft_x[max_y_index],fft_y[max_y_index]),
           xytext=(fft_x[max_y_index]+20,fft_y[max_y_index]-0.33),fontsize=15)
        ax0[i,2].plot(fft_x,fft_y,'black')
        ax0[i,0].set_yticks([])
        ax0[i,1].set_yticks([])
        ax0[i,2].set_yticks([])
        if i<period_num-1:
           ax0[i,0].set_xticks([])
           ax0[i,1].set_xticks([])
        plt.setp(ax0[i,0].xaxis.get_majorticklabels(), rotation=20)
    ax0[i,1].set_xlabel('Frequency(MHz)',fontsize=25)
    ax0[i,2].set_xlabel('Frequency(Hz)',fontsize=25)
    ax0[i,0].set_xlabel('Time (UTC)',fontsize=25) 
    plt.savefig('6periodic.png',dpi=100) 
    
def plot_classified_sporad(data):
    f_arr=(np.arange(3200)+400)/4
    sporad_data = data
    sporad_com = sporad_data['rfi_component']
    sporad_base = sporad_data['rfi_basis']
    sporad_finfo = sporad_data['f_info']/4
    sporad_num0 = sporad_com.shape[0]
    weights=[va_imp_sum/va_sum,va_period_sum/va_sum,va_sporad_sum/va_sum]
    if sporad_num0 == 0:
        pass
    if sporad_num0 >10:
        sporad_num=10
    else: 
        sporad_num=sporad_num0
    fig0, ax0 = plt.subplots(sporad_num, 2, sharex=False, sharey=True,figsize=(20, 12))
    fig0.subplots_adjust(hspace=0,wspace=0,left = 0.01,right = 0.99,bottom = 0.13,top = 0.8)
    ax1=fig0.add_axes([0.01, 0.85, 0.8,0.145])
    ax3=fig0.add_axes([0.28, 0.83, 0.34,0.16])
    ax2=fig0.add_axes([0.65, 0.83, 0.34,0.16])
    ax2.tick_params(axis="x", labelsize=15)
    ax3.tick_params(axis="x", labelsize=15)
    ax2.tick_params(axis="y", labelsize=15)
    ax3.tick_params(axis="y", labelsize=15)
    for sporad_finfo_ in sporad_finfo:
        f_disy = np.zeros(4096*4)+sporad_finfo_ ############
        ax3.plot(t_arr1,f_disy,c='green')
    ax1.axis('off')
    text_type = 'RFI type:'
    text_type_ = 'Non-stationary'
    text_com = 'Component Number: %s'%(sporad_num0)
    text_va = 'Sum of Variance-Ratio: %1.1f%%'%(weights[2]*100)
    ax1.text(0,0.75,text_type,fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0.1,0.75,text_type_,color='green',fontdict=font,fontsize=20,weight='heavy')
    ax1.text(0,0.45,text_com,fontdict=font,fontsize=20)
    ax1.text(0,0.15,text_va,fontdict=font,fontsize=20)
    va_x = np.arange(len(va))+1
    ax2.plot(va_x,np.log(va),'.',linewidth=0.5,color='black')
    for ele in np.log(list(va_sporad)):
        ax2.scatter(va_x[np.log(va)==ele],ele,color='green',s=50)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    ax2.text(len(va)/5,1.8*max(np.log(va)),'Variance-Ratio of RFI Weights',fontdict=font,fontsize=15,weight='light')
    print('Non-stationary:%s'%(sporad_num))
    for i in range(sporad_num):
        plot_data0 = (sporad_com[i]-min(sporad_com[i]))/(max(sporad_com[i])-min(sporad_com[i]))
        ax0[i,0].plot(t_arr1,plot_data0,'black')
        plot_data1 = (sporad_base[i]-min(sporad_base[i]))/(max(sporad_base[i])-min(sporad_base[i]))
        ax0[i,1].plot(f_arr,plot_data1,'black')
        ax0[i,1].scatter(sporad_finfo[i],plot_data1[int(sporad_finfo[i]*4)-400],s=50,color='green')
        f_node_ = 'Frequency: '
        if sporad_finfo[i] >750:
           text_x =  sporad_finfo[i]-300
        elif sporad_finfo[i] <=200:
           text_x =  sporad_finfo[i]+200
        else:
           text_x =  sporad_finfo[i]+50 
        if plot_data1[int(sporad_finfo[i]*4)-400]>0:
            text_y = np.mean(plot_data1)+0.3
        if plot_data1[int(sporad_finfo[i]*4)-400]<=0:
            text_y = np.mean(plot_data1)-0.3    
        ax0[i,1].annotate("%s%s MHz"%(f_node_,sporad_finfo[i]),xy=(sporad_finfo[i],plot_data1[int(sporad_finfo[i]*4)-400]),
           xytext=(text_x,text_y),fontsize=15)
        plt.setp(ax0[i,0].xaxis.get_majorticklabels(), rotation=20)
        ax0[i,0].set_yticks([])
        ax0[i,1].set_yticks([])
        if i<sporad_num-1:
           ax0[i,0].set_xticks([])
           ax0[i,1].set_xticks([]) 
    ax0[i,1].set_xlabel('Frequency(MHz)',fontsize=25) 
    ax0[i,0].set_xlabel('Time (UTC)',fontsize=25) 
    plt.savefig('6nonsta.png',dpi=100) 
plot_classified_imp(imp_data)
plot_classified_period(period_data)
#plot_classified_sporad(sporad_data)
