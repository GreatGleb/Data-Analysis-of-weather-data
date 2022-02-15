#! /usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------
# Эта программа использует способы анализа данных Python
#
# (C) 2021 Сугак Глеб, Винница, Украина
# email: gwelbts@gmail.com
# -----------------------------------------------------------

from pylab import *
import random
import time
import numpy as np
import pandas as pd
from pandas.plotting import scatter_matrix
import webbrowser
from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
plt.style.use('seaborn')
import matplotlib.cm as cm
import matplotlib
import matplotlib as mpl
import matplotlib.colors as colors
import matplotlib.dates
from matplotlib.dates import (YEARLY, DateFormatter,
                              rrulewrapper, RRuleLocator, drange)
import datetime
from datetime import datetime
from datetime import timedelta
import sys
import seaborn as sns
import math
from tabulate import tabulate
from sklearn.metrics import r2_score

pd.options.mode.chained_assignment = None  # default='warn'
pd.options.display.max_rows = 5
pd.options.display.max_columns = 30
pd.options.display.width = 1750

def is_number(value):
	if(is_float(value)):
		if(pd.isna(float(value)) == 0):
			return True
	else:
		return False

def is_float(value):
  try:
  	float(value)
  	return True
  except:
    return False

NAME_OF_FILE = "weatherAUS.csv"
datasetWeather = pd.read_csv(NAME_OF_FILE)
# print(datasetWeather)

base_html = """
<!doctype html>
<html><head>
<meta http-equiv="Content-type" content="text/html; charset=utf-8">
<script type="text/javascript" src="https://ajax.googleapis.com/ajax/libs/jquery/2.2.2/jquery.min.js"></script>
<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.16/css/jquery.dataTables.css">
<script type="text/javascript" src="https://cdn.datatables.net/1.10.16/js/jquery.dataTables.js"></script>
</head><body>%s<script type="text/javascript">$(document).ready(function(){$('table').DataTable({
    "pageLength": 50
});});</script>
</body></html>
"""

def df_html(df):
    """HTML table with pagination and other goodies"""
    df_html = df.to_html()
    return base_html % df_html

def df_window(df):
    """Open dataframe in browser window using a temporary file"""
    with NamedTemporaryFile(delete=False, suffix='.html', mode="w") as f:
        f.write(df_html(df))
    webbrowser.open(f.name)


def pirson(x, y):
#     return random.uniform(0, 1)
	iX = 0
	sumX = 0
	for val in x:
		if(is_number(val)):
			sumX = sumX+float(val)
		iX = iX+1
	meanX = sumX/iX
	# print(meanX, " ", sumX, " ", iX)

	sum_diffX = 0

	if(meanX > 0):
		for val in x:
			if(is_number(val)):
				sum_diffX = sum_diffX + (meanX-float(val))*(meanX-float(val))
		# print(sum_diffX)
		# print("\n")

	iY = 0
	sumY = 0
	for val in y:
		if(is_number(val)):
			sumY = sumY+float(val)
		iY = iY+1
	meanY = sumY/iY
	# print(meanY, " ", sumY, " ", iY)

	if((sumX > 0 and sumY > 0) != 1):
		return "False"

	sum_diffY = 0

	for val in y:
		if(is_number(val)):
			sum_diffY = sum_diffY + ((meanY-float(val))*(meanY-float(val)))
	# print(sum_diffY)
	# print("\n")

	sum_meandiff = 0

	i_XY = 0
	for valX in x:
		valY = y[i_XY]
		a = meanX
		if(is_number(valX)):
			a = a - float(valX)
		b = meanY
		if(is_number(valY)):
			b = b - float(valY)

		sum_meandiff = sum_meandiff + (a*b)
		i_XY = i_XY+1
	# print(sum_meandiff)

	pirson = sum_meandiff/math.sqrt(sum_diffX*sum_diffY)

	if pirson > 1:
		pirson = 1.0

	return pirson

column_list = []
header_list = []
value_list = []
n = 0

# for key in datasetWeather.keys():
# 	value_list.append([])
# for key in datasetWeather.keys():
# 	if(n>1):
# 	    isNeedToAddKey = 0
# 	    m = 0
# 	    for j in datasetWeather.keys():
# 	        coeff_pirs = pirson(datasetWeather[key], datasetWeather[j])
# 	        if(coeff_pirs != ""):
# 	            if(m>1 and m != n):
# 	                value_list[m-2].append(coeff_pirs)
# 	            if(m ==n):
# 	                value_list[m-2].append(1)
# 	            m = m+1
# 	            isNeedToAddKey = 1
# 	    if(isNeedToAddKey == 1):
# 		    column_list.append(key)
# 		    header_list.append(key)
# 	n = n+1

# print(value_list)
# print(column_list)
# print(header_list)

value_list = [[1, 0.7685952974707007, 0.10504524886001243, 0.41991641325159657, 0.16864229585582036, 'False', 0.16920057070487207, 'False', 'False', 0.18623702760471705, 0.1853533735079125, -0.1555139641618754, 0.02397724894316705, 0.4890344162454526, 0.4827002909819153, 0.1982170863211145, 0.15486093582003102, 0.9453658692306607, 0.7172317863505138, 'False', 'False'], [0.7685952974707007, 1, -0.07012688909594861, 0.46351326417801136, 0.3116396651799017, 'False', 0.1262236221755322, 'False', 'False', 0.025857197729950508, 0.07363081517482331, -0.40495960512799084, -0.4528256333093518, 0.47223464519991565, 0.4665090884744154, -0.11586147147287207, -0.09947948848108425, 0.9404201216684095, 1.0, 'False', 'False'], [0.10504524886001243, -0.07012688909594861, 1, -0.061555092501703525, -0.11915284972673862, 'False', 0.11489814529259308, 'False', 'False', 0.08558917774510928, 0.05434919985416703, 0.22261528101757586, 0.24495259163160035, -0.0019510486247615063, -0.0019059840473415828, 0.17930637894718235, 0.15295671585145149, 0.01308116838348214, -0.08028735520342349, 'False', 'False'], [0.41991641325159657, 0.46351326417801136, -0.061555092501703525, 1, 0.7133111860698167, 'False', 0.19028767268854901, 'False', 'False', 0.1972261805124905, 0.1536763019282916, -0.39544959412282277, -0.3105053037358524, 0.7881501474674704, 0.7870059341386509, 0.21250629838970822, 0.23804015909249482, 0.46631719344024314, 0.4510163861849289, 'False', 'False'], [0.16864229585582036, 0.3116396651799017, -0.11915284972673862, 0.7133111860698167, 1, 'False', 0.08654776800433919, 'False', 'False', 0.12799270943350188, 0.16146659981710137, -0.3292824660691454, -0.32818248169539294, 1.0, 1.0, 0.0915487527019434, 0.12058637723260744, 0.2670980588631997, 0.3757717217097279, 'False', 'False'], ['False', 'False', 'False', 'False', 'False', 1, 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False'], [0.16920057070487207, 0.1262236221755322, 0.11489814529259308, 0.19028767268854901, 0.08654776800433919, 'False', 1, 'False', 'False', 0.6645027728979473, 0.7887847591821369, -0.20479511193844369, 0.007303298179325541, 0.7058624006577002, 0.7121542251372889, 0.01899045524026636, 0.0953337517929356, 0.16421153767518118, 0.16422335762006463, 'False', 'False'], ['False', 'False', 'False', 'False', 'False', 'False', 'False', 1, 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False'], ['False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 1, 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False'], [0.18623702760471705, 0.025857197729950508, 0.08558917774510928, 0.1972261805124905, 0.12799270943350188, 'False', 0.6645027728979473, 'False', 'False', 1, 0.5473232425557523, -0.23656990036572698, -0.011909020079785209, 0.4209197317652407, 0.41418615724059915, 0.12276044679368409, 0.14762599333238266, 0.14590819269443253, 0.029174457725561955, 'False', 'False'], [0.1853533735079125, 0.07363081517482331, 0.05434919985416703, 0.1536763019282916, 0.16146659981710137, 'False', 0.7887847591821369, 'False', 'False', 0.5473232425557523, 1, -0.12066820928831948, 0.09795708630632813, 0.6440601348859848, 0.6730447861495167, 0.10828907052521397, 0.13093549503569124, 0.17956644056477158, 0.12869838100415795, 'False', 'False'], [-0.1555139641618754, -0.40495960512799084, 0.22261528101757586, -0.39544959412282277, -0.3292824660691454, 'False', -0.20479511193844369, 'False', 'False', -0.23656990036572698, -0.12066820928831948, 1, 0.7523320362452861, 0.16062144466170106, 0.12902884656598568, 0.3257376632278232, 0.22781932757519416, -0.3464044891364969, -0.3955120011815731, 'False', 'False'], [0.02397724894316705, -0.4528256333093518, 0.24495259163160035, -0.3105053037358524, -0.32818248169539294, 'False', 0.007303298179325541, 'False', 'False', -0.011909020079785209, 0.09795708630632813, 0.7523320362452861, 1, 0.16680939880260515, 0.2023924125857876, 0.3510779963077419, 0.41435406029789756, -0.19063573194475214, -0.3612530736874266, 'False', 'False'], [0.4890344162454526, 0.47223464519991565, -0.0019510486247615063, 0.7881501474674704, 1.0, 'False', 0.7058624006577002, 'False', 'False', 0.4209197317652407, 0.6440601348859848, 0.16062144466170106, 0.16680939880260515, 1, 1.0, 0.6301909996164945, 0.7057749755927073, 0.5368815690262355, 0.6050783755759849, 'False', 'False'], [0.4827002909819153, 0.4665090884744154, -0.0019059840473415828, 0.7870059341386509, 1.0, 'False', 0.7121542251372889, 'False', 'False', 0.41418615724059915, 0.6730447861495167, 0.12902884656598568, 0.2023924125857876, 1.0, 1, 0.6330920797999336, 0.7187156338621193, 0.507733148979441, 0.6354554369889852, 'False', 'False'], [0.1982170863211145, -0.11586147147287207, 0.17930637894718235, 0.21250629838970822, 0.0915487527019434, 'False', 0.01899045524026636, 'False', 'False', 0.12276044679368409, 0.10828907052521397, 0.3257376632278232, 0.3510779963077419, 0.6301909996164945, 0.6330920797999336, 1, 1.0, 0.03118784590953467, -0.13583243412014984, 'False', 'False'], [0.15486093582003102, -0.09947948848108425, 0.15295671585145149, 0.23804015909249482, 0.12058637723260744, 'False', 0.0953337517929356, 'False', 'False', 0.14762599333238266, 0.13093549503569124, 0.22781932757519416, 0.41435406029789756, 0.7057749755927073, 0.7187156338621193, 1.0, 1, 0.040960106014510896, -0.06419161157862449, 'False', 'False'], [0.9453658692306607, 0.9404201216684095, 0.01308116838348214, 0.46631719344024314, 0.2670980588631997, 'False', 0.16421153767518118, 'False', 'False', 0.14590819269443253, 0.17956644056477158, -0.3464044891364969, -0.19063573194475214, 0.5368815690262355, 0.507733148979441, 0.03118784590953467, 0.040960106014510896, 1, 0.8910137084093162, 'False', 'False'], [0.7172317863505138, 1.0, -0.08028735520342349, 0.4510163861849289, 0.3757717217097279, 'False', 0.16422335762006463, 'False', 'False', 0.029174457725561955, 0.12869838100415795, -0.3955120011815731, -0.3612530736874266, 0.6050783755759849, 0.6354554369889852, -0.13583243412014984, -0.06419161157862449, 0.8910137084093162, 1, 'False', 'False'], ['False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 1, 'False'], ['False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 'False', 1], [], []]
column_list = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']
header_list = ['MinTemp', 'MaxTemp', 'Rainfall', 'Evaporation', 'Sunshine', 'WindGustDir', 'WindGustSpeed', 'WindDir9am', 'WindDir3pm', 'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm', 'Pressure9am', 'Pressure3pm', 'Cloud9am', 'Cloud3pm', 'Temp9am', 'Temp3pm', 'RainToday', 'RainTomorrow']

value_list.insert(0, header_list)

# maxNumOfValues = 0
# for value in value_list:
#     if maxNumOfValues < len(value):
#         maxNumOfValues = len(value)
#
# i =  0
# for value in value_list:
#     diff = 0
#     if len(value) < maxNumOfValues:
#         diff = maxNumOfValues-len(value)
#     for k in range(diff):
#         value_list[i].append("")
#     i = i+1

column_list.insert(0, "Name")

# print(len(header_list))
# print('len(header_list)')
# for value in value_list:
#     print(len(value))

objForCorr = {}
n = 0
for column in column_list:
    objForCorr[column] = value_list[n]
    n = n+1

pdForCorr = pd.DataFrame(objForCorr)
# df_window(pdForCorr)

def pirsonForecast(location, column1, column2):
#     print(datasetWeather)
    dates = []
    values1 = []
    values2 = []
    i = -1
    for loc in datasetWeather[datasetWeather.keys()[1]]:
        i = i+1
        if loc == location and datasetWeather[column2][i] < 22 and datasetWeather[column2][i] > 5:
            dates.append(datetime.strptime(datasetWeather[datasetWeather.keys()[0]][i], '%Y-%m-%d'))
            values1.append(datasetWeather[column1][i])
            values2.append(datasetWeather[column2][i])

    forecastValues = []
    period = 10
    for i in range(period):
        forecastValues.append(None)

    for v in range(len(values1)):
        if v < period:
            continue
#         if v > 50:
#             break
        meanForLastDays = 0
        for i in range(period):
            if is_number(values2[v-i-1]) and is_number(values1[v-i-1]):
                meanForLastDays += values2[v-i-1]-values1[v-i-1]
        meanForLastDays = meanForLastDays/period
#         print('klkl')
#         print(meanForLastDays)
        forV = values1[v]+meanForLastDays
        forecastValues.append(forV)

    mean_error = 0
    i = 0
    for value in values2:
        f = forecastValues[i]
        if is_number(f) and is_number(value):
            error = abs(f - value)
            mean_error = mean_error + error
        i = i + 1
    mean_error = mean_error/len(values2)

#     print(forecastValues)
    print("mean_error")
    print(mean_error)

    plt.plot_date(dates, values1, ms=2, color='green')
    plt.plot_date(dates, values2, ms=2, color='red')
    plt.plot_date(dates, forecastValues, ms=2, color='blue')

    plt.xlabel("Time")
    plt.ylabel("Temperature at 9 am")
    plt.show()
#     print(dates)

pirsonForecast('Albury', 'MinTemp', 'Temp9am')

# Рисунок первый
# Диаграма рассеивания

def showScatter(x, y, labelX, labelY, color):
	fig, ax = plt.subplots()
	ax.scatter(x, y, c = color, s=0.5)
	ax.legend("Correlation of MinTemp and Temp9am")
	plt.ylabel(labelY)
	plt.xlabel(labelX)
	plt.show()

# showScatter(datasetWeather[datasetWeather.keys()[2]], datasetWeather[datasetWeather.keys()[19]], datasetWeather.keys()[2], datasetWeather.keys()[19], "#ad09a3")

def int_r(num):
    num = int(num + (0.5 if num > 0 else -0.5))
    return num

def showGistogramByWindDir(x, y, labelX, labelY):
    newX = []
    newY = []
    i = -1
    for loc in datasetWeather[datasetWeather.keys()[1]]:
        i = i+1
        if loc == 'Albury':
            newX.append(x[i])
            newY.append(y[i])
    x = newX
    y = newY

    minY = 0
    maxY = 0
    sumY = 0
    for j in y:
        if(j < minY):
            minY = j
        if(j > maxY):
            maxY = j
        if is_number(j):
            sumY = sumY + j

    meanY = sumY/len(y)

    variance = 0
    for j in y:
        if is_number(j):
            variance = variance + (j-meanY)**2
    variance = math.sqrt(variance/len(y))

    interval = 3.49*variance/(len(y)**(1/3))

    classesX = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE', 'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
	# for j in x:
	# 	if j not in classesX:
	# 		if pd.isna(j) == 0:
	# 			classesX.append(j)
    #
    intervals = []
    i = 0
    while 1:
        a = minY+(i*interval)
        b = a+interval
        if b >= maxY:
            b = maxY
        intervals.append([a, b])
        if b >= maxY:
            break
        i = i + 1

    arrayValues = {}
    for i in classesX:
        arrayValues[i] = []
        for j in range(len(intervals)):
            arrayValues[i].append(0)

    m = 0
    for j in x:
        if(pd.isna(j) == 0 and pd.isna(y[m]) == 0):
            l = 0
            for i in intervals:
                if(l == len(intervals)-1):
                    if(y[m] >= i[0]):
                        break
                if(y[m] >= i[0] and y[m] < i[1]):
                    break
                l = l+1
            arrayValues[j][l] = arrayValues[j][l]+1
        m = m+1

    print(arrayValues)

    indexes = []
    indexes2 = []
    for k in intervals:
        digit1 = int_r(k[0]*100)/100
        digit2 = int_r(k[1]*100)/100
        indexes.append("[" + str(digit1) + ", " + str(digit2) + ")")
        indexes2.append(str(digit1) + "-" + str(digit2))

    probabilitiesOld = {}
    probabilities = {}
    for index in indexes2:
        probabilitiesOld[index] = {}
        probabilities[index] = {}

    sumsOfDiapazones = 0

    for obj in arrayValues:
        for val in arrayValues[obj]:
            sumsOfDiapazones = sumsOfDiapazones + val
#     print(sumsOfDiapazones)
    for obj in arrayValues:
        i = 0
        for val in arrayValues[obj]:
            probabilitiesOld[indexes2[i]][str((float(val)/sumsOfDiapazones))] = obj
#             print(val)
#             print(sumsOfDiapazones[i])
            i = i+1
#     print(probabilitiesOld)

    arrayProp = []
    for p in probabilitiesOld:
        for pp in probabilitiesOld[p]:
            arrayProp.append(float(pp))
    arrayProp.sort()

    sumP = 0
    for p in arrayProp:
        for po in probabilitiesOld:
            if str(p) in probabilitiesOld[po]:
                probabilities[po][str(sumP) + '-' + str(sumP+p)] = probabilitiesOld[po][str(p)]
#                 print(str(sumP) + '-' + str(sumP+p))
                sumP = sumP + p

#     print(probabilities)

    intervalsOriginal = []
    intervals = []
    for p in probabilities:
        intervalsOriginal.append(p.split('-'))
    for interv in intervalsOriginal:
        intervals_arr = []
        for inter in interv:
            intervals_arr.append(inter)
        intervals.append(intervals_arr)

    precisionOfForecast_sum = 0
    i = 0
    for valY in y:
        for interval in intervals:
            if valY > float(interval[0]) and valY <= float(interval[1]):
                arr = probabilities[str(interval[0])+'-'+str(interval[1])]
                arrOriginal = []
                for a in arr:
                    arrOriginal.append(a.split('-'))
                forecastValue = ''
                while forecastValue == '':
                    randomNumber = random.uniform(0, 1)
                    for a in arrOriginal:
                        if randomNumber > float(a[0]) and randomNumber <= float(a[1]):
                            forecastValue = arr[a[0]+'-'+a[1]]
                if forecastValue == x[i]:
                    precisionOfForecast_sum = precisionOfForecast_sum + 1
        i = i +1
    print(precisionOfForecast_sum)
    print(len(y))

    df = pd.DataFrame(arrayValues, index=indexes)
    ax = df.plot.bar(rot=0)
    plt.xticks(rotation=65)
    plt.ylabel(labelY)
    plt.xlabel(labelX)

    plt.show()

# showGistogramByWindDir(datasetWeather[datasetWeather.keys()[9]], datasetWeather[datasetWeather.keys()[13]], datasetWeather.keys()[13], "Number of cases")

# print(random.uniform(0, 20))

def showPolinomRegression(location, column1, column2):
    dates = []
    values1 = []
    values2 = []
    i = -1
    for loc in datasetWeather[datasetWeather.keys()[1]]:
        i = i+1
        if loc == location and datasetWeather[column2][i] < 22 and datasetWeather[column2][i] > 5:
            dates.append(datetime.strptime(datasetWeather[datasetWeather.keys()[0]][i], '%Y-%m-%d'))
            values1.append(datasetWeather[column1][i])
            values2.append(datasetWeather[column2][i])

    forecastValues = []
    period = 10
    for i in range(period):
        forecastValues.append(None)

    for v in range(len(values1)):
        if v < period:
            continue
#         if v > 50:
#             break
        meanForLastDays = 0
        for i in range(period):
            if is_number(values2[v-i-1]) and is_number(values1[v-i-1]):
                meanForLastDays += values2[v-i-1]-values1[v-i-1]
        meanForLastDays = meanForLastDays/period
#         print('klkl')
#         print(meanForLastDays)
        forV = values1[v]+meanForLastDays
        forecastValues.append(forV)

#     plt.plot_date(dates, values1, ms=2, color='green')
#     plt.plot_date(dates, values2, ms=2, color='red')
#     plt.plot_date(dates, forecastValues, ms=2, color='blue')

    y = []
    dates2 = []
    x = []
    i = 0
    for value in dates:
        if is_number(values2[i]) and is_number(values1[i]):
            y.append(values2[i])
            x.append(values1[i]) #dates[i].timestamp()
            dates2.append(dates[i])
#         if is_number(values2[i]) != True:
#             y.append(0)
#             x.append(0)

        i = i + 1

#     y.sort()
#     x.sort()
    y = np.array(y)
    x = np.array(x)

#     print(values1)

    p5 = np.poly1d(np.polyfit(x,y,20))
    p5x = p5(x)

    meanError = 0

    period = 10
    i = 0
    for val in p5x:
        if i > period:
            mean = 0
            for j in range(period-1):
                mean = mean + (p5x[i-j-2]-p5x[i-j-1])
            mean = mean/period

            forecast = y[i-1]+mean*7
            meanError = meanError+(forecast-y[i])
        i = i+1

    meanError = meanError/(len(p5x)-period)
    print('meanError')
    print(meanError)
    print(len(p5x)-period)

    p5xnumpy = []

    for p in p5x:
        p5xnumpy.append(p)

    expression1s = []
    i = 0
    for val in y:
        expression1s.append(abs(val-p5x[i])/val)

    plt.plot_date(dates, values2, ms=2,color='blue')
    plt.plot_date(dates, values1, ms=2,color='green')
    plt.plot_date(dates2, p5x, ms=2,color='red')
#     p5x.sort()
#     x.sort()

#     plt.plot(x, p5x, c='red')
#     plt.scatter(values1,values2, c = '#0800a3', s=5)

#     fig, ax = plt.subplots()
#     ax.scatter(x, y, c = '#43e56a', s=1)

    approximationError = sum(expression1s)/len(y)*100

    print(approximationError)
    print(x[0])
    print(y[0])
    print(p5x[0])

    plt.xlabel("Minimum Temperature")
    plt.ylabel("Temperature at 9 am")
    plt.show()

showPolinomRegression('Albury', 'MinTemp', 'Temp9am')