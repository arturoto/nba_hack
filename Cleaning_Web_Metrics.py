import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

attendenceData = pd.read_csv('Attendance_and_Capacity_Data.csv')
attendenceData['CA_Ratio'] = attendenceData['Attendance'] / attendenceData['Capacity']

webMetrics = pd.read_csv('Web_Metrics.csv')
team_abv = pd.read_csv('Team_abbreviations.csv')


ATL_WebM = webMetrics[(webMetrics.Team == 'ATL') & (webMetrics.Metric == 'UNQ')]
ATL_TTS_WebM = webMetrics[(webMetrics.Team == 'ATL') & (webMetrics.Metric == 'TTS')]

ATL_WebM['ATL_TTS1'] = ATL_TTS_WebM.Metric.values
ATL_WebM['ATL_TTS'] = ATL_TTS_WebM.Value.values

webM_f = pd.DataFrame()

webM_f = ATL_WebM
webM_f = webM_f.rename(columns = {'Team': 'team_ATL'})
webM_f = webM_f.rename(columns = {'Metric': 'ATL_UNQ1'})
webM_f = webM_f.rename(columns = {'Value': 'ATL'})


webM_f = webM_f.drop('ATL_TTS1', 1)
webM_f = webM_f.drop('team_ATL', 1)
webM_f = webM_f.drop('ATL_UNQ1', 1)


teamList = list(team_abv.iloc[:, 0])
#print(teamList)

for team in teamList:
	if team == 'ATL':
		continue
	#webM_f['team_'+str(team)] = team

	valAdd = webMetrics[(webMetrics.Team == str(team)) & (webMetrics.Metric == 'UNQ')]
	#webM_f[str(team)+'_UNQ'] = valAdd.Metric.values
	webM_f[str(team)] = valAdd.Value.values

	valAddTTS = webMetrics[(webMetrics.Team == str(team)) & (webMetrics.Metric == 'TTS')]
	#webM_f[str(team)+'_TTS'] = valAddTTS.Metric.values
	webM_f[str(team)+'_TTS'] = valAddTTS.Value.values


#print(webM_f.head())
webM_f['sort'] = pd.to_datetime(webM_f['Date'])
webM_f = webM_f.sort_values(by='sort', ascending=True)
webM_f = webM_f.drop('sort', 1)


for teamA in teamList:
	for teamB in teamList:
		webM_f[str(teamA) + ',' + str(teamB)] = webM_f[str(teamA)] + webM_f[str(teamB)]

for teamA in teamList:
	for teamB in teamList:
		webM_f[str(teamB)+ ',' + str(teamA)] = webM_f[str(teamA)] + webM_f[str(teamB)]


webM_f.to_csv('Web_Metrics_b.csv')

print(webM_f.describe())