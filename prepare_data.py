# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:57:45 2019

@author: mleummens
Структура текстовых выходных файлов из WinRiver (// - разделитель запятая):
    строка: время(год,месяц,число,часов,минут,секунд,сотых секунд)//расстояние от ADCP//скорость восток//скорость север//
    //смещение отностительно ВТ на восток//смещение относительно ВТ на север//пройденной расстояние по ВТ//глубина//скорость абс//направление//
    //широта//долгота(если есть данные GPS) 
    всего 10 элементов и 9 разделителей (с GPS 12 и 11)
"""

print('Запуск программы для обработки данных ADCP...')
import time
t0 = time.time()
import sys
import os
dir = os.path.dirname(__file__)
import numpy as np
import math
import csv
import pandas as pd
from scipy import interpolate
import pickle

class structtype(): # Определить класс structure как в MATLAB
    pass

def round_half_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def findGridDistStepGPS(N,len,data):
    mean_de = np.empty(N); mean_de[:] = np.nan 
    mean_dn = np.empty(N); mean_dn[:] = np.nan 
    mean_dist = np.empty(N); mean_dist[:] = np.nan 
    for kk in range(0,N):
        delta_de = np.empty(len[kk]-1); delta_de[:] = np.nan 
        delta_dn = np.empty(len[kk]-1); delta_dn[:] = np.nan
        for ii in range(0,len[kk]-1):
            delta_de[ii] = data[kk].ref_de[ii+1]-data[kk].ref_de[ii]
            delta_dn[ii] = data[kk].ref_dn[ii+1]-data[kk].ref_dn[ii]
        mean_de[kk] = np.nanmean(delta_de)
        mean_dn[kk] = np.nanmean(delta_dn)
        mean_dist[kk] = np.sqrt((mean_de[kk]**2)+(mean_dn[kk]**2))
    grid_dist_step = round_half_up(np.nanmean(mean_dist),1)
    return grid_dist_step

def findGridDistStepBT(N,len,data):
    mean_de = np.empty(N); mean_de[:] = np.nan 
    mean_dn = np.empty(N); mean_dn[:] = np.nan 
    mean_dist = np.empty(N); mean_dist[:] = np.nan 
    for kk in range(0,N):
        delta_de = np.empty(len[kk]-1); delta_de[:] = np.nan 
        delta_dn = np.empty(len[kk]-1); delta_dn[:] = np.nan
        for ii in range(0,len[kk]-1):
            delta_de[ii] = data[kk].BT_de[ii+1]-data[kk].BT_de[ii]
            delta_dn[ii] = data[kk].BT_dn[ii+1]-data[kk].BT_dn[ii]
        mean_de[kk] = np.nanmean(delta_de)
        mean_dn[kk] = np.nanmean(delta_dn)
        mean_dist[kk] = np.sqrt((mean_de[kk]**2)+(mean_dn[kk]**2))
    grid_dist_step = round_half_up(np.nanmean(mean_dist),1)
    return grid_dist_step
    
def movingAverage2D(data,hwin,vwin):
    out = np.empty((data.shape[0],data.shape[1])); out[:] = np.nan 
    for ii in range(0+hwin,data.shape[0]-2*hwin):
        for jj in range(0+vwin,data.shape[1]-2*vwin):
            out[ii,jj] = np.mean(data[ii-hwin:ii+hwin,jj-vwin:jj+vwin])
    return out

def getAngleBetweenVectorAndAxis(de,dn,dist,v_base,n):
    v = np.empty((dist.shape[0],3)); v[:] = np.nan
    mod = np.empty((dist.shape[0],3)); mod[:] = np.nan
    dot = np.empty(dist.shape[0]); dot[:] = np.nan         
    det = np.empty(dist.shape[0]); det[:] = np.nan     
    ang = np.empty(dist.shape[0]); ang[:] = np.nan     
    for ii in range(n,dist.shape[0]-n):
        v[ii,:] = np.asarray([de[ii+n]-de[ii-n],dn[ii+n]-dn[ii-n],0])
    v[:,0] = np.interp(dist,dist[~np.isnan(v[:,0])],v[~np.isnan(v[:,0]),0])         
    v[:,1] = np.interp(dist,dist[~np.isnan(v[:,1])],v[~np.isnan(v[:,1]),1])  
    mod = np.sqrt((v[:,0]**2)+(v[:,1]**2))               
    for ii in range(0,dist.shape[0]):
        v[ii,:] = v[ii,:]/mod[ii]
    dot = v_base[0]*v[:,0]+v_base[1]*v[:,1]
    det = v_base[0]*v[:,1]-v_base[1]*v[:,0]
    ang = np.degrees(np.arctan2(det,dot))
    return ang

def calculateDepthAverage(v,z):
    DA_v = np.empty(v.shape[0]); DA_v[:] = np.nan 
    for ii in range(0,v.shape[0]):
        temp_v = v[ii,~np.isnan(v[ii,:])]
        temp_z = z[ii,~np.isnan(v[ii,:])]
        if temp_v.shape[0] > 1:
            Av = np.empty(temp_v.shape[0]-1); Av[:] = np.nan
            for jj in range(temp_v.shape[0]-1):
                Av[jj] = 0.5*(temp_v[jj]+temp_v[jj+1])*(temp_z[jj+1]-temp_z[jj])
            DA_v[ii] = np.sum(Av)/(temp_z[-1]-temp_z[0])
    return DA_v

def interpolateDataOnGrid(data,dist,z,base_interp_grid_dist,base_interp_grid_z):
    # замена координат сетки на 'nan' где нет данных
    interp_grid_dist = np.copy(base_interp_grid_dist);
    interp_grid_z = np.copy(base_interp_grid_z);
    max_z = np.nanmax(z,1)
    interp_max_z = np.interp(base_interp_grid_dist[:,0],dist,max_z) 
    I = np.ones((base_interp_grid_dist.shape[0],base_interp_grid_dist.shape[1]),dtype=bool)
    for ii in range(0,interp_grid_dist.shape[0]):
        if np.isnan(interp_max_z[ii]) == True:
            I[ii,:] = 0
        else:
            I[ii,:] = interp_grid_z[ii,:] < interp_max_z[ii]
    interp_grid_dist[I == False] = 'nan'
    interp_grid_z[I == False] = 'nan'
    del I
    # создание входных файлов для функции интерполяции        
    temp_length_z = z.shape[1];
    temp_z = np.reshape(z,np.product(z.shape))
    temp_dist = np.empty(temp_z.shape[0]); temp_dist[:] = np.nan 
    for ii in range(0,data.shape[0]):  
        temp_dist[ii*temp_length_z:(ii+1)*temp_length_z-1] = dist[ii]
    values = np.reshape(data,np.product(z.shape))
    values = values[~np.isnan(temp_z)]; temp_dist = temp_dist[~np.isnan(temp_z)]; temp_z = temp_z[~np.isnan(temp_z)]
    values = values[~np.isnan(temp_dist)]; temp_z = temp_z[~np.isnan(temp_dist)]; temp_dist = temp_dist[~np.isnan(temp_dist)] 
    points = np.vstack((temp_dist, temp_z)); points = points.transpose()
    # интерполяция
    out = interpolate.griddata(points, values, (interp_grid_dist, interp_grid_z), method='linear')
    return out, interp_grid_dist, interp_grid_z
 
#%% Входные данные обработка
N = 74                  # Количество трансект
GPS = 0;                # Есть ли данные GPS: 0 - нет, используется трек дна, 1 - есть, используются GPS координаты
if GPS == 0: variables = 9 # Количество разделителей в строке в текстовом файле
elif GPS == 1: variables = 11
start_bank = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]        # Стартовый берег трансекты: левый - 0, правый - 1  
dist_correction = [50,90,90,55,105,70,75,50,65,60,80,60,80,65,80,65,70,65,90,70,75,70,70,65,80,60,70,70,90,65,\
    60,60,100,55,75,60,75,65,70,60,75,65,50,70,65,60,65,55,70,65,85,60,65,55,75,65,70,60,60,55,75,70,80,\
    70,85,60,75,60,70,55,70,60,65,55]   # Доп. расстояние для совмещения поперечников по глубине
ANGLE = -30 # Угол доминирующего направления течения на отливе относительно базового вектора (на север) от -180 до 180, - по часовой стрелке , + против часовой (смотреть по данным)
grid_dist_step = 0      # Шаг сетки по поперечнику для интерполяции данных в метрах
                        # Если 0, то определяется автоматически как медиана расстояний между измеренными точками    
nh = 8          # Количество ячеек с каждой стороны от точки для сглаживания данных по горизонтали 
nv = 5          # Количество ячеек с каждой стороны от точки для сглаживания данных по вертикали
                # Если оба значения не равны 0 - никакого сглаживания
bank = 0; # Основной берег (от которого измеряется расстояние): 0 - левый, 1 - правый
orientation = 1; # направление отлива относительно среднего поперечника (0 - -90 градусов, 1 - +90 градусов)
ZONE = 37 # Зона UTM
profile_correction = 0 # 0 - если средний поперечник определяется автоматически; 1 - если вводится вручную

#%% Чтение данных из исходных текстовых файлов 
print('Чтение данных из исходных текстовых файлов...')
filename = []
transect = [structtype() for i in range(N)]
length = np.zeros(N,dtype=int) 
for kk in range(0,N):
    filename.append('txt/template_bt_new_{:03d}_ASC.TXT'.format(kk))            # Переменное название входного файла
    with open(os.path.join(dir, filename[kk])) as csvDataFile:
        data = list(csv.reader(csvDataFile,delimiter = ','))
    ind = np.zeros((len(data),variables),dtype=int)
    L = np.zeros(len(data),dtype=int)
    for ii in range(0,len(data)):                                               
        ind[ii,:] = np.asarray([i for i, value in enumerate(data[ii]) if value == ''], dtype=int)     # Индекс пустых ячеек (разделитель данных)
        L[ii] = ind[ii,1]-ind[ii,0]-1                                           # Количество точек со скоростями для каждого момента времени
    length[kk] = len(data)
    transect[kk].timedata = np.zeros((length[kk],7),dtype=int)
    transect[kk].z = np.empty((length[kk],max(L))); transect[kk].z[:] = np.nan 
    transect[kk].ve = np.empty((length[kk],max(L))); transect[kk].ve[:] = np.nan 
    transect[kk].vn = np.empty((length[kk],max(L))); transect[kk].vn[:] = np.nan
    transect[kk].depth = np.empty(length[kk]); transect[kk].depth[:] = np.nan 
    transect[kk].mag = np.empty((length[kk],max(L))); transect[kk].mag[:] = np.nan 
    transect[kk].dir = np.empty((length[kk],max(L))); transect[kk].dir[:] = np.nan
    for ii in range(0,length[kk]):                                                       
        transect[kk].timedata[ii,:] = np.asarray(data[ii][0:7], dtype=int)                         # Время       
        transect[kk].z[ii,0:L[ii]] = np.asarray(data[ii][ind[ii,0]+1:ind[ii,1]], dtype=float)      # Расстояние от ADCP
        transect[kk].ve[ii,0:L[ii]] = np.asarray(data[ii][ind[ii,1]+1:ind[ii,2]], dtype=float)     # Скорость на восток
        transect[kk].vn[ii,0:L[ii]] = np.asarray(data[ii][ind[ii,2]+1:ind[ii,3]], dtype=float)     # Скорость на север
        transect[kk].depth[ii] = np.asarray(data[ii][ind[ii,6]+1], dtype=float)                    # Глубина
        transect[kk].mag[ii,0:L[ii]] = np.asarray(data[ii][ind[ii,7]+1:ind[ii,8]], dtype=float)    # Скорость абсолютная
        transect[kk].dir[ii,0:L[ii]] = np.asarray(data[ii][ind[ii,8]+1:ind[ii,8]+1+L[ii]], dtype=float)    # Направление течения 
    if GPS == 1:
        transect[kk].xcoord = np.empty(length[kk]); transect[kk].xcoord[:] = np.nan 
        transect[kk].ycoord = np.empty(length[kk]); transect[kk].ycoord[:] = np.nan      
        for ii in range(0,length[kk]):                                                       
            transect[kk].ycoord[ii] = np.asarray(data[ii][ind[ii,9]+1], dtype=float)                   # Долгота 
            transect[kk].xcoord[ii] = np.asarray(data[ii][ind[ii,10]+1], dtype=float)                  # Широта    
    
    df = pd.DataFrame({'year':  transect[kk].timedata[:,0]+2000,\
                       'month': transect[kk].timedata[:,1],\
                       'day':   transect[kk].timedata[:,2],\
                       'hour':  transect[kk].timedata[:,3],\
                       'minute': transect[kk].timedata[:,4],\
                       'second': transect[kk].timedata[:,5],\
                       'ms':    transect[kk].timedata[:,6]*10})
    transect[kk].time = pd.to_datetime(df) 
    transect[kk].time_mean = transect[kk].time[0]+(transect[kk].time[len(data)-1]-transect[kk].time[0])/2
    ### Координаты относительно ВТ
    transect[kk].BT_de = np.empty(length[kk]); transect[kk].BT_de[:] = np.nan 
    transect[kk].BT_dn = np.empty(length[kk]); transect[kk].BT_dn[:] = np.nan
    transect[kk].BT_dist = np.empty(length[kk]); transect[kk].BT_dist[:] = np.nan
    for ii in range(0,length[kk]):
        transect[kk].BT_de[ii] = np.asarray(data[ii][ind[ii,3]+1], dtype=float)                       # Смещение лодки на восток
        transect[kk].BT_dn[ii] = np.asarray(data[ii][ind[ii,4]+1], dtype=float)                       # Смещение лодки на север
        transect[kk].BT_dist[ii] = np.asarray(data[ii][ind[ii,5]+1], dtype=float)                     # Пройденное расстояние        
    del ind, L, df
del filename, ii, kk, data
      
#%% Обработка данных базовая
for kk in range(0,N):
    ### Замена точек со значением скорости -32768 на NaN
    transect[kk].z[transect[kk].ve == -32768] = 'nan'               
    transect[kk].ve[transect[kk].ve == -32768] = 'nan'
    transect[kk].vn[transect[kk].vn == -32768] = 'nan'
    transect[kk].mag[transect[kk].mag == -32768] = 'nan'
    transect[kk].dir[transect[kk].dir == -32768] = 'nan'
    ### Удалить точки где по всей ветикали нет данных    
    I = ~np.all(np.isnan(transect[kk].z), axis=1)
    transect[kk].depth = transect[kk].depth[I]
    transect[kk].ve = transect[kk].ve[I,:]; transect[kk].vn = transect[kk].vn[I,:]
    transect[kk].mag = transect[kk].mag[I,:]; transect[kk].dir = transect[kk].dir[I,:]
    transect[kk].time = transect[kk].time[I]; transect[kk].timedata = transect[kk].timedata[I,:]
    transect[kk].z = transect[kk].z[I,:]
    length[kk] = len(transect[kk].depth)
    transect[kk].BT_de = transect[kk].BT_de[I]; transect[kk].BT_dn = transect[kk].BT_dn[I]; transect[kk].BT_dist = transect[kk].BT_dist[I];   
    if GPS == 1:
        transect[kk].xcoord[transect[kk].xcoord == -32768] = 'nan'
        transect[kk].ycoord[transect[kk].ycoord == -32768] = 'nan'
        transect[kk].xcoord = transect[kk].xcoord[I]; transect[kk].ycoord = transect[kk].ycoord[I]    
    del I            
    ### Приводка всех измерений к старту от правого берега (если нет данных GPS)
    if start_bank[kk] == 0:
        temp_de = np.zeros(len(transect[kk].BT_de))
        temp_dn = np.zeros(len(transect[kk].BT_dn))    
        temp_de[0] = transect[kk].BT_de[-1]
        temp_dn[0] = transect[kk].BT_dn[-1]
        for ii in range(0,len(transect[kk].BT_de)-1):
            temp_de[ii+1] = transect[kk].BT_de[-1]-(transect[kk].BT_de[-1]-transect[kk].BT_de[-1-ii])
            temp_dn[ii+1] = transect[kk].BT_dn[-1]-(transect[kk].BT_dn[-1]-transect[kk].BT_dn[-1-ii])
        temp_dist = np.sqrt(np.add(np.multiply(temp_de[:],temp_de[:]),np.multiply(temp_dn[:],temp_dn[:])))
        transect[kk].BT_de = temp_de; transect[kk].BT_dn = temp_dn; transect[kk].BT_dist = temp_dist                
        del temp_de, temp_dn, temp_dist
        ### Добавка дополнительного расстояния для совмещения поперечников
        transect[kk].BT_dist = transect[kk].BT_dist+dist_correction[kk]
        ### Изменение основного берега с правого на левый
        if bank == 0:           
            transect[kk].BT_de = np.nanmax(transect[kk].BT_de)-transect[kk].BT_de
            transect[kk].BT_dn = np.nanmax(transect[kk].BT_dn)-transect[kk].BT_dn
            transect[kk].BT_dist = np.nanmax(transect[kk].BT_dist)-transect[kk].BT_dist
        del ii 
    elif start_bank[kk] == 1:
        transect[kk].BT_de = -1*transect[kk].BT_de[-1]+transect[kk].BT_de
        transect[kk].BT_dn = -1*transect[kk].BT_dn[-1]+transect[kk].BT_dn
        transect[kk].BT_dist = np.sqrt((transect[kk].BT_de**2)+(transect[kk].BT_dn**2))
del kk
    
#%% Определение среднего поперечника по координатам и проекция данных
if GPS == 0:
    ### Трансекты, которые начинались от берега, противоположного тому, на котором находится базовая точка, развернуть чтобы расстояние шло в возрастающем порядке 
    for kk in range(0,N):
        if transect[kk].BT_dist[0] > transect[kk].BT_dist[-1]:
            transect[kk].depth = np.flip(transect[kk].depth)
            transect[kk].mag = np.flipud(transect[kk].mag); transect[kk].dir = np.flipud(transect[kk].dir)
            transect[kk].ve = np.flipud(transect[kk].ve); transect[kk].vn = np.flipud(transect[kk].vn)
            transect[kk].z = np.flipud(transect[kk].z); 
            transect[kk].time = np.flip(transect[kk].time); transect[kk].timedata = np.flipud(transect[kk].timedata)
            transect[kk].BT_dist = np.flip(transect[kk].BT_dist)
            transect[kk].BT_de = np.flip(transect[kk].BT_de); transect[kk].BT_dn = np.flip(transect[kk].BT_dn)
elif GPS == 1:
    ### Удалить данные в точках, где нет координат
    for kk in range(0,N):
        I = ~np.isnan(transect[kk].xcoord)
        transect[kk].depth = transect[kk].depth[I]
        transect[kk].ve = transect[kk].ve[I,:]; transect[kk].vn = transect[kk].vn[I,:]
        transect[kk].mag = transect[kk].mag[I,:]; transect[kk].dir = transect[kk].dir[I,:]
        transect[kk].time = transect[kk].time[I]; transect[kk].timedata = transect[kk].timedata[I,:]
        transect[kk].z = transect[kk].z[I,:]
        transect[kk].xcoord = transect[kk].xcoord[I]; transect[kk].ycoord = transect[kk].ycoord[I]
        length[kk] = len(transect[kk].depth)
        transect[kk].BT_de = transect[kk].BT_de[I]; transect[kk].BT_dn = transect[kk].BT_dn[I]; transect[kk].BT_dist = transect[kk].BT_dist[I];   
        del I  
    ### Перевод координат из широты и долготы в метры UTM (нужна среда pyproj)
    from pyproj import Proj
    p = Proj(proj='utm',zone=ZONE,ellps='WGS84', preserve_units=False)
    for kk in range(0,N):
        transect[kk].xcoord_proj = np.empty(len(transect[kk].xcoord)); transect[kk].xcoord_proj[:] = np.nan 
        transect[kk].ycoord_proj = np.empty(len(transect[kk].ycoord)); transect[kk].ycoord_proj[:] = np.nan      
        transect[kk].xcoord_proj,transect[kk].ycoord_proj = p(transect[kk].xcoord,transect[kk].ycoord)
        transect[kk].xcoord_proj[transect[kk].xcoord_proj == 1e+30] = 'nan'
        transect[kk].ycoord_proj[transect[kk].ycoord_proj == 1e+30] = 'nan'
    ### Определение среднего поперечника методом наименьших квадратов
    length_total = np.sum(length)
    temp_xcoord = np.empty(length_total); temp_xcoord[:] = np.nan
    temp_ycoord = np.empty(length_total); temp_ycoord[:] = np.nan
    for kk in range(0,N):    # Все точки по х и по у помещаются в один вектор
        temp_xcoord[np.sum(length[0:kk+1])-length[kk]:np.sum(length[0:kk+1])] = transect[kk].xcoord_proj;
        temp_ycoord[np.sum(length[0:kk+1])-length[kk]:np.sum(length[0:kk+1])] = transect[kk].ycoord_proj;
    temp_xcoord = temp_xcoord[~np.isnan(temp_xcoord)]
    temp_ycoord = temp_ycoord[~np.isnan(temp_ycoord)]
    temp_A = np.vstack([temp_xcoord, np.ones(len(temp_xcoord))]).T
    m, c = np.linalg.lstsq(temp_A, temp_ycoord, rcond=None)[0]
    xplot = np.asarray([np.min(temp_xcoord), np.max(temp_xcoord)])
    yplot = m*xplot+c
    del temp_xcoord, temp_ycoord, temp_A
    if profile_correction == 1:
        xplot = [458250, 459500]
        yplot = [7314750, 7316100]
        m = (yplot[0]-yplot[1])/(xplot[0]-xplot[1])
        c = yplot[1] - m*xplot[1]
    ### Проекция данных на средний поперечник
    for kk in range(0,N):
        transect[kk].xcoord_proj_cs = np.empty(len(transect[kk].xcoord_proj)); transect[kk].xcoord_proj_cs[:] = np.nan 
        transect[kk].ycoord_proj_cs = np.empty(len(transect[kk].ycoord_proj)); transect[kk].ycoord_proj_cs[:] = np.nan 
        transect[kk].xcoord_proj_cs = (transect[kk].xcoord_proj - m*c + m*transect[kk].ycoord_proj)/(m**2+1)
        transect[kk].ycoord_proj_cs = (c + m*transect[kk].xcoord_proj + (m**2)*transect[kk].ycoord_proj)/(m**2+1)
    del m, c
    ### Смещение и пройденное расстояние по поперечнику
    if bank == 0:
        start_point = [xplot[0],yplot[0]]
        end_point = [xplot[1],yplot[1]]
    elif bank == 1:
        start_point = [xplot[1],yplot[1]]
        end_point = [xplot[0],yplot[0]]  
    for kk in range(0,N):
        transect[kk].ref_de = np.empty(len(transect[kk].xcoord_proj_cs)); transect[kk].ref_de[:] = np.nan
        transect[kk].ref_dn = np.empty(len(transect[kk].ycoord_proj_cs)); transect[kk].ref_dn[:] = np.nan
        transect[kk].ref_dist = np.empty(len(transect[kk].xcoord_proj_cs)); transect[kk].ref_dist[:] = np.nan
        if bank == 0:   
            transect[kk].ref_de = transect[kk].xcoord_proj_cs - start_point[0]
            transect[kk].ref_dn = transect[kk].ycoord_proj_cs - start_point[1]
        elif bank == 1:
            transect[kk].ref_de = start_point[0] - transect[kk].xcoord_proj_cs
            transect[kk].ref_dn = start_point[1] - transect[kk].ycoord_proj_cs    
        transect[kk].ref_dist = np.sqrt((transect[kk].ref_de**2)+(transect[kk].ref_dn**2))
    ### Трансекты, которые начинались от берега, противоположного тому, на котором находится базовая точка, развернуть чтобы расстояние шло в возрастающем порядке 
    for kk in range(0,N):
        if transect[kk].ref_dist[0] > transect[kk].ref_dist[-1]:
            transect[kk].ref_dist = np.flip(transect[kk].ref_dist)
            transect[kk].ref_de = np.flip(transect[kk].ref_de); transect[kk].ref_dn = np.flip(transect[kk].ref_dn)
            transect[kk].depth = np.flip(transect[kk].depth)
            transect[kk].mag = np.flipud(transect[kk].mag); transect[kk].dir = np.flipud(transect[kk].dir)
            transect[kk].ve = np.flipud(transect[kk].ve); transect[kk].vn = np.flipud(transect[kk].vn)
            transect[kk].z = np.flipud(transect[kk].z); 
            transect[kk].time = np.flip(transect[kk].time); transect[kk].timedata = np.flipud(transect[kk].timedata)
            transect[kk].xcoord = np.flip(transect[kk].xcoord); transect[kk].ycoord = np.flip(transect[kk].ycoord)
            transect[kk].xcoord_proj = np.flip(transect[kk].xcoord_proj); transect[kk].ycoord_proj = np.flip(transect[kk].ycoord_proj)
            transect[kk].xcoord_proj_cs = np.flip(transect[kk].xcoord_proj_cs); transect[kk].ycoord_proj_cs = np.flip(transect[kk].ycoord_proj_cs)
            transect[kk].BT_dist = np.flip(transect[kk].BT_dist)
            transect[kk].BT_de = np.flip(transect[kk].BT_de); transect[kk].BT_dn = np.flip(transect[kk].BT_dn)
    t1 = time.time()
    print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1
       

#%% Коррекция направления течения и определение отливной/приливной скорости
v_base = np.asarray([0,1,0])    # угол относительно базового вектора (на север) от -180 до 180, - по часовой стрелке , + против часовой
if GPS == 1:
    print('Коррекция направления течения...')
    np.warnings.filterwarnings('ignore') 
    ### Спроецировать скорости относительно ВТ на поперечник по GPS координатам
    for kk in range(0,N):
        transect[kk].GGA_de = transect[kk].xcoord_proj-transect[kk].xcoord_proj[0]
        transect[kk].GGA_dn = transect[kk].ycoord_proj-transect[kk].ycoord_proj[0]
        transect[kk].GGA_dist = np.sqrt((transect[kk].GGA_de**2)+(transect[kk].GGA_dn**2))    
        ### Определить курс судна по GPS и по BT с помощью трека
        transect[kk].angGGA = getAngleBetweenVectorAndAxis(transect[kk].GGA_de,transect[kk].GGA_dn,transect[kk].GGA_dist,v_base,3)
        transect[kk].angBT = getAngleBetweenVectorAndAxis(transect[kk].BT_de,transect[kk].BT_dn,transect[kk].BT_dist,v_base,3)
        ### Определить направление течения по BT в той же системе координат v_base       
        dot = np.add(v_base[0]*transect[kk].ve,v_base[1]*transect[kk].vn)
        det = np.add(v_base[0]*transect[kk].vn,-1*v_base[1]*transect[kk].ve)
        transect[kk].dir = np.degrees(np.arctan2(det,dot))
        del dot, det 
        
        ### Определить направление течения относительно курса судна по BT    
        transect[kk].phi = np.empty((transect[kk].dir.shape[0],transect[kk].dir.shape[1])); transect[kk].phi[:] = np.nan
        transect[kk].GGA_dir = np.empty((transect[kk].dir.shape[0],transect[kk].dir.shape[1])); transect[kk].GGA_dir[:] = np.nan    
        for ii in range(0,length[kk]):
            transect[kk].phi[ii,:] = transect[kk].dir[ii,:]-transect[kk].angBT[ii] % 360
            I = np.where(transect[kk].phi[ii,:] >= 180)
            transect[kk].phi[ii,I] = transect[kk].phi[ii,I] - 360
            I = np.where(transect[kk].phi[ii,:] < -180)
            transect[kk].phi[ii,I] = transect[kk].phi[ii,I] + 360
            ### Определить направление течения относительно географических координат, перенеся разницу между курсом судна и направлением течения по ВТ
            ### на курс по данным GPS     
            transect[kk].GGA_dir[ii,:] = transect[kk].angGGA[ii]+transect[kk].phi[ii,:] % 360
            I = np.where(transect[kk].GGA_dir[ii,:] >= 180)
            transect[kk].GGA_dir[ii,I] = transect[kk].GGA_dir[ii,I] - 360
            I = np.where(transect[kk].GGA_dir[ii,:] < 180)
            transect[kk].GGA_dir[ii,I] = transect[kk].GGA_dir[ii,I] + 360
        ### Спроецировать абсолютную скорость на оси север/восток в соответствии с полученным направлением течения в географических координатах
        transect[kk].GGA_vn = transect[kk].mag*np.cos(np.radians(transect[kk].GGA_dir))
        transect[kk].GGA_ve = transect[kk].mag*-1*np.sin(np.radians(transect[kk].GGA_dir))    
        del I
    
    t1 = time.time()
    print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1

#%% Постреоние сетки и интерполяция данных
print('Интерполяция данных на сетку...')
### найти максимальную глубину для определения сетки по z
max_depth = np.empty(N); max_depth[:] = np.nan 
dist_BT = np.empty(N); dist_BT[:] = np.nan 
for kk in range(0,N):
    max_depth[kk] = np.nanmax(transect[kk].z)
    dist_BT[kk] =  np.nanmax(transect[kk].BT_dist)
### найти шаг сетки по поперечнику (по расстоянию от начальной точки)
if grid_dist_step == 0:
    if GPS == 0: 
        grid_dist_step = findGridDistStepBT(N,length,transect)
        dist_max = np.nanmax(dist_BT)
    elif GPS == 1: 
        grid_dist_step = findGridDistStepGPS(N,length,transect)
        dist_max = np.sqrt(((xplot[1]-xplot[0])**2)+((yplot[1]-yplot[0])**2))    
### построение общей сетки и интерполяция
base_interp_grid_dist, base_interp_grid_z = np.mgrid[0:dist_max:grid_dist_step, 0.25:np.nanmax(max_depth):0.1]
grid_dist = np.arange(0,dist_max,grid_dist_step)
grid_z = np.arange(0,np.nanmax(max_depth),0.1)

transect_grid = [structtype() for i in range(N)]
for kk in range(0,N):
    if GPS == 0:
        [transect_grid[kk].ve, transect_grid[kk].interp_grid_dist, transect_grid[kk].interp_grid_z] = interpolateDataOnGrid(\
            transect[kk].ve,transect[kk].BT_dist,transect[kk].z,base_interp_grid_dist,base_interp_grid_z)
        [transect_grid[kk].vn, transect_grid[kk].interp_grid_dist, transect_grid[kk].interp_grid_z] = interpolateDataOnGrid(\
            transect[kk].vn,transect[kk].BT_dist,transect[kk].z,base_interp_grid_dist,base_interp_grid_z)        
        transect_grid[kk].depth = np.interp(grid_dist,transect[kk].BT_dist,transect[kk].depth,left=-999,right=-999)
    elif GPS == 1:
        [transect_grid[kk].ve, transect_grid[kk].interp_grid_dist, transect_grid[kk].interp_grid_z] = interpolateDataOnGrid(\
            transect[kk].GGA_ve,transect[kk].ref_dist,transect[kk].z,base_interp_grid_dist,base_interp_grid_z)
        [transect_grid[kk].vn, transect_grid[kk].interp_grid_dist, transect_grid[kk].interp_grid_z] = interpolateDataOnGrid(\
            transect[kk].GGA_vn,transect[kk].ref_dist,transect[kk].z,base_interp_grid_dist,base_interp_grid_z)
        transect_grid[kk].depth = np.interp(grid_dist,transect[kk].ref_dist,transect[kk].depth,left=-999,right=-999)
    transect_grid[kk].depth[transect_grid[kk].depth == -999] = 'nan'
    transect_grid[kk].mag = np.sqrt((transect_grid[kk].ve**2)+(transect_grid[kk].vn**2))
    dot = np.add(v_base[0]*transect_grid[kk].ve,v_base[1]*transect_grid[kk].vn)
    det = np.add(v_base[0]*transect_grid[kk].vn,-1*v_base[1]*transect_grid[kk].ve)
    transect_grid[kk].dir = np.degrees(np.arctan2(det,dot))
t1 = time.time()
print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1

#%% Определение перпендикуляра к среднему поперечнику для получения приливной/отливной компоненты скорости
print('Определение приливной/отливной скорости...')
if GPS == 0:
    angle = ANGLE
elif GPS == 1:
    v_data = np.asarray([end_point[0]-start_point[0],end_point[1]-start_point[1],0])
    mod = np.sqrt((v_data[0]**2)+(v_data[1]**2))
    v_data = v_data/mod
    dot = v_base[0]*v_data[0]+v_base[1]*v_data[1]
    det = v_base[0]*v_data[1]-v_base[1]*v_data[0]
    angle = np.degrees(np.arctan2(det,dot))
    if orientation == 0:
        angle = angle-90 % 360;
    elif orientation == 1:
        angle = angle+90 % 360;
    if angle >= 180:
        angle = angle - 360
    elif angle < -180:
        angle = angle + 360
    ### получить географические координаты точек сетки
    profile_xcoord = start_point[0]+grid_dist*np.cos(np.radians(angle))
    profile_ycoord = start_point[1]+grid_dist*np.sin(np.radians(angle))
    
### Вычислить приливные/отливные и поперечные скорости
for kk in range(0,N):
    ### Угол между направлением течения и направлением отлива
    transect_grid[kk].delta = transect_grid[kk].dir-angle % 360
    for ii in range(0,grid_dist.shape[0]):    
        I = np.where(transect_grid[kk].delta[ii,:] >= 180)
        transect_grid[kk].delta[ii,I] = transect_grid[kk].delta[ii,I] - 360
        I = np.where(transect_grid[kk].delta[ii,:] < -180)
        transect_grid[kk].delta[ii,I] = transect_grid[kk].delta[ii,I] + 360
    ### Скорости (v1 - отлив/прилив, v2 - поперечные скорости, положительные значения на -90 градусов от направления отливного вектора)    
    transect_grid[kk].v1 = transect_grid[kk].mag*np.cos(np.radians(transect_grid[kk].delta))
    transect_grid[kk].v2 = transect_grid[kk].mag*-1*np.sin(np.radians(transect_grid[kk].delta))    
np.warnings.resetwarnings()     

t1 = time.time()
print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1
            
#%% Осреднение (сглаживание) данных
print('Сглаживание данных...')
transect_filter = [structtype() for i in range(N)]
for kk in range(0,N):    
    transect_filter[kk].ve = movingAverage2D(transect_grid[kk].ve,nh,nv)
    transect_filter[kk].vn = movingAverage2D(transect_grid[kk].vn,nh,nv)
    transect_filter[kk].mag = np.sqrt((transect_filter[kk].ve**2)+(transect_filter[kk].vn**2))
    dot = np.add(v_base[0]*transect_filter[kk].ve,v_base[1]*transect_filter[kk].vn)
    det = np.add(v_base[0]*transect_filter[kk].vn,-1*v_base[1]*transect_filter[kk].ve)
    transect_filter[kk].dir = np.degrees(np.arctan2(det,dot))

    ### Угол между направлением течения и направлением отлива
    transect_filter[kk].delta = transect_filter[kk].dir-angle % 360
    for ii in range(0,grid_dist.shape[0]):    
        I = np.where(transect_filter[kk].delta[ii,:] >= 180)
        transect_filter[kk].delta[ii,I] = transect_filter[kk].delta[ii,I] - 360
        I = np.where(transect_filter[kk].delta[ii,:] < -180)
        transect_filter[kk].delta[ii,I] = transect_filter[kk].delta[ii,I] + 360
    transect_filter[kk].v1 = transect_filter[kk].mag*np.cos(np.radians(transect_filter[kk].delta))
    transect_filter[kk].v2 = transect_filter[kk].mag*-1*np.sin(np.radians(transect_filter[kk].delta))    

t1 = time.time()
print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1
   
#%% Осреднение скоростей течения по глубине
print('Осреднение скоростей течения по глубине...')
transect_DA = [structtype() for i in range(N)]
### Средние по глубине скорости в географических координатах (север/восток)
for kk in range(0,N):       
    transect_DA[kk].ve = calculateDepthAverage(transect_filter[kk].ve,base_interp_grid_z)
    transect_DA[kk].vn = calculateDepthAverage(transect_filter[kk].vn,base_interp_grid_z)
    transect_DA[kk].mag = np.sqrt((transect_DA[kk].ve**2)+(transect_DA[kk].vn**2))
    dot = np.add(v_base[0]*transect_DA[kk].ve,v_base[1]*transect_DA[kk].vn)
    det = np.add(v_base[0]*transect_DA[kk].vn,-1*v_base[1]*transect_DA[kk].ve)
    transect_DA[kk].dir = np.degrees(np.arctan2(det,dot))

### Средние по глубине приливные/отливные и поперечные скорости  
for kk in range(0,N):       
    transect_DA[kk].v1 = calculateDepthAverage(transect_grid[kk].v1,base_interp_grid_z)
    transect_DA[kk].v2 = calculateDepthAverage(transect_grid[kk].v2,base_interp_grid_z)           

t1 = time.time()
print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1

#%% Сохранение данных в файл
#outputDataFile = open('figures/figure_input1.pickle',"wb+")
#pickle.dump(transect_filter,outputDataFile,protocol=0)
#outputDataFile.close()
#
#outputDataFile = open('figures/figure_input.pickle',"r")
#pickle.load(outputDataFile)
#outputDataFile.close()


#%% Входные данные рисунки
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

graph = 1; # Строить (1) или не строить (0) графики
xlb = 'Расстояние от левого берега[м]'        
if bank == 1:
    xlb = 'Расстояние от правого берега[м]'
F = 18
spx = 3 # Количество рисунков на одном изображении по x 
spy = 2 # Количество рисунков на одном изображении по y 

#%% Построение графиков необработанных данных
if graph == 1:
    n = 74
    if GPS == 1:
        print('Построение графиков...')
        ### Треки поперечников
        fig = plt.figure(figsize=(9,9), dpi = 100)
        for kk in range(0,n):
            plt.plot(transect[kk].xcoord_proj, transect[kk].ycoord_proj, label='{}'.format(kk))
        plt.plot(xplot, yplot, 'k', label='Fitted line', linewidth=2.5)
        plt.legend(loc='upper right',ncol=2)
        plt.grid(True)
        plt.legend()
        plt.xlabel('X [м]')
        plt.ylabel('Y [м]')
        plt.xlim(458000,460500)
        plt.ylim(7314000,7316500)
        fig.savefig(os.path.join(dir, 'figures/profile_tracks.png'), bbox_inches = 'tight', dpi = 150)
        del kk
    
    ### Поперечные профили глубина
    fig = plt.figure(figsize=(9,4.5), dpi = 100)
    if GPS == 0:
        for kk in range(0,n):
            plt.plot(transect[kk].BT_dist,transect[kk].depth,label='{}'.format(kk))        
    elif GPS == 1:
        for kk in range(0,n):
            plt.plot(transect[kk].ref_dist,transect[kk].depth,label='{}'.format(kk))
    plt.legend(loc='upper right',ncol=2)
    plt.grid(True)
    plt.xlabel(xlb)
    plt.ylabel('Глубина [м]')
    ax = plt.axes()
    ax.xaxis.set_major_locator(plticker.MultipleLocator(100))
    ax.xaxis.set_minor_locator(plticker.MultipleLocator(25))
    fig.savefig(os.path.join(dir, 'figures/depth_profile.png'), bbox_inches = 'tight', dpi = 150)
    del kk
    
    ### Скорости течения поперечники
    subplots = spx*spy
    parts = math.ceil(N/subplots)
    splits = np.zeros(parts+1,dtype=int) # разделение поперечников на разные рисунки
    splits[0] = 0
    temp = 1
    for ii in range(0,parts):
        splits[ii+1] = temp/parts*N
        temp = temp + 1
    del temp
    
    x = 0; y = 0 # создание векторов для определения положения поперечников на рисунках
    nxplot = np.zeros(subplots,dtype=int); nyplot = np.zeros(subplots,dtype=int)
    for ii in range(0,subplots):
        nxplot[ii] = x; nyplot[ii] = y
        if x+1 <= spx-1:
            x = x+1; y = y
        else:
            x = 0; y = y+1
    del x, y    
        
    for ii in range (0,parts): # построение рисунков
        fig,axes = plt.subplots(spy,spx,figsize=(24,12), dpi = 100, constrained_layout=True)
        nn = 0
        for kk in range(splits[ii],splits[ii+1]):
            axes[nyplot[nn],nxplot[nn]].set_axisbelow(True)           
            if GPS == 0:
                axes[nyplot[nn],nxplot[nn]].plot(transect[kk].BT_dist,-1*transect[kk].depth,'k')                
            elif GPS == 1:
                axes[nyplot[nn],nxplot[nn]].plot(transect[kk].ref_dist,-1*transect[kk].depth,'k')
            plt1 = axes[nyplot[nn],nxplot[nn]].contourf(transect_grid[kk].interp_grid_dist,-1*transect_grid[kk].interp_grid_z,transect_filter[kk].v1,\
                       15,cmap='jet') # Данные, интерполированные на сетку
            plt1.set_clim(-0.5, 0.5)
            cbar1 = plt.colorbar(plt1,ax=axes[nyplot[nn],nxplot[nn]])
            cbar1.set_label('Скорость течения [м/с]',fontsize=F)
            axes[nyplot[nn],nxplot[nn]].set_title('tr{} {} '.format(kk,transect[kk].time_mean),fontsize=F)
            axes[nyplot[nn],nxplot[nn]].set_xlabel(xlb)
            axes[nyplot[nn],nxplot[nn]].set_ylabel('Расстояние от ADCP [м]',fontsize=F)
            axes[nyplot[nn],nxplot[nn]].set_xlim(-50, 350)
            axes[nyplot[nn],nxplot[nn]].set_ylim(-15, 0)
            nn = nn+1
        plt.rcParams['axes.grid'] = True
        plt.tick_params(axis='both', labelsize=F)
        fig.savefig(os.path.join(dir, 'figures/raw_vel_magnitude_transect_{}_{}_grid.png'.format(splits[ii]+1,splits[ii+1])),\
                    bbox_inches = 'tight', dpi = 150)
    
    ### Графики осредненных по глубине скоростей течения
    if GPS == 1:
        M = 5
        for ii in range (0,parts): # построение рисунков
            fig,axes = plt.subplots(spy,spx,figsize=(24,12), dpi = 100, constrained_layout=True)        
            nn = 0
            for kk in range(splits[ii],splits[ii+1]):
                axes[nyplot[nn],nxplot[nn]].set_axisbelow(True)
                axes[nyplot[nn],nxplot[nn]].plot(xplot, yplot, 'k', linewidth=1)
                plt1 = axes[nyplot[nn],nxplot[nn]].quiver(profile_xcoord[::M],profile_ycoord[::M],transect_DA[kk].ve[::M],transect_DA[kk].vn[::M],\
                    transect_DA[kk].mag[::M], scale = 1/0.15, cmap='jet')
                plt1.set_clim(0, 2)
                cbar1 = plt.colorbar(plt1,ax=axes[nyplot[nn],nxplot[nn]])
                cbar1.set_label('Скорость течения [м/с]',fontsize=F)
                axes[nyplot[nn],nxplot[nn]].quiverkey(plt1,0.9,0.1,1,'1 m/s', coordinates='axes')
                axes[nyplot[nn],nxplot[nn]].set_title('tr{} {} '.format(kk,transect[kk].time_mean),fontsize=F)
                axes[nyplot[nn],nxplot[nn]].set_xlim(458000,460500)
                axes[nyplot[nn],nxplot[nn]].set_ylim(7314000,7316500)
                axes[nyplot[nn],nxplot[nn]].set_xlabel('X [м]',fontsize=F)
                axes[nyplot[nn],nxplot[nn]].set_ylabel('Y [м]',fontsize=F)
                axes[nyplot[nn],nxplot[nn]].set_aspect('equal')        
                nn = nn+1            
            plt.rcParams['axes.grid'] = True
            fig.savefig(os.path.join(dir, 'figures/DA_transect_{}_{}_grid.png'.format(splits[ii]+1,splits[ii+1])),\
                        bbox_inches = 'tight', dpi = 150)
    
    ### Графики эпюр в самой глубокой точке для характерных моментов
    fig = plt.figure(figsize=(6,6), dpi = 100)
    iplot = np.arange(0,n,4)
    ind = 219
    for kk in range(0,len(iplot)):
        plt.plot(transect_filter[iplot[kk]].v1[ind,:],transect[iplot[kk]].depth[ind]-transect_grid[iplot[kk]].interp_grid_z[ind,:],\
                 label='tr{} {} '.format(kk,transect[kk].time_mean))        
    #plt.legend(loc='upper right',ncol=1)
    plt.xlabel('Скорость течения [м/с]')
    plt.ylabel('Расстояние от дна [м]')
    plt.xlim(-0.5,0.5)
    plt.ylim(0,12)
    plt.grid(True)

    fig.savefig(os.path.join(dir, 'figures/profiles.png'.format(splits[ii]+1,splits[ii+1])),\
                bbox_inches = 'tight', dpi = 150)


#    del axes, nn, parts, subplots, splits
#    del nxplot, nyplot
#del graph, xlb, spx, spy
#del xplot, yplot
#del bank 
    t1 = time.time()
    print('Время:{} c'.format(round(t1-t0,2))); t0 = t1; del t1
    
#%% Экспорт данных
export_dist_grid = grid_dist
export_time = []
export_DA_v1 = np.zeros((len(grid_dist),N))
export_v1_profile_ind = np.zeros((np.size(base_interp_grid_z,1),N))
export_depth_profile_ind = np.zeros(N)
for kk in range(0,N):
    export_DA_v1[:,kk] = transect_DA[kk].v1
    export_time.append(str(transect[kk].time_mean))
    export_v1_profile_ind[:,kk] = transect_filter[kk].v1[ind,:]
    export_depth_profile_ind[kk] = transect_grid[kk].depth[ind]
