import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



gameData = pd.read_csv('game_data copy.csv')
viewershipData = pd.read_csv('Viewership.csv')
webMetricData = pd.read_csv('Web_Metrics_b.csv')
standardData = pd.DataFrame()

# Adding Viewship data
viewershipData['sort'] = pd.to_datetime(viewershipData['DATE'])
viewershipData = viewershipData.sort_values(by='sort', ascending=True)
viewershipData = viewershipData.drop('sort', 1)

standardData['Date'] = pd.to_datetime(viewershipData['DATE'])
standardData['Away'] = viewershipData['Away']
standardData['Home'] = viewershipData['HOME']

viewsSeries = viewershipData['Total Viewership']
viewScaleFactor = 10/viewsSeries.max()

standardData['Scale_Views'] = viewershipData['Total Viewership'] * viewScaleFactor


#Adding game data

gameData['Date'] = pd.to_datetime(gameData['Date'])

#standardData['Percent_Tickets_Sold'] = 0

#standardData['Percent_tickets_Sold'] = (gameData['Date'] == standardData['Date']) * 3

#standardData.merge(gameData['Date'])

#print(standardData.head())

standardData = pd.merge(standardData, gameData[['Date', 'Home', 'Percent_Tickets_Sold']], \
																	on=['Date', 'Home'])

standardData['Percent_Tickets_Sold'] = standardData['Percent_Tickets_Sold'] * 10 


# Merging Game Data
standardData = pd.merge(standardData, gameData[['Date', 'Home', 'Total_All_Stars']], \
																	on=['Date', 'Home'])
allStarScaler = 10 / standardData['Total_All_Stars'].max()
standardData['Total_All_Stars'] = standardData['Total_All_Stars'] * allStarScaler

standardData = pd.merge(standardData, gameData[['Date', 'Home', 'Weekend', 'Weekday']], \
																	on=['Date', 'Home'])


# Merging website visit
webMetricData['Date'] = pd.to_datetime(webMetricData['Date'])

standardData = pd.merge(standardData, webMetricData[['Date', 'ATL', 'BOS', 'BRK', 'CHA', \
													'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GSW', \
													'HOU', 'IND', 'LAC', 'LAL', 'MEM', 'MIA', \
													'MIL', 'MIN', 'NOP', 'NYK', 'OKC', 'ORL', \
													'PHI', 'PHX', 'POR', 'SAC', 'SAS', 'TOR', \
													'UTA', 'WAS']], on='Date')



viewsData = standardData.iloc[:, 8:]


#betting the list
team_abv = pd.read_csv('Team_abbreviations.csv')
teamList = list(team_abv.iloc[:, 0])

homeList = list(standardData['Home'])
awayList = list(standardData['Away'])
cList = []

for i, elem in enumerate(homeList):
	cList.append(str(homeList[i])+ ',' + str(awayList[i]))


for teamA in teamList:
	for teamB in teamList:
		viewsData[str(teamA) + ',' + str(teamB)] = viewsData[str(teamA)] + viewsData[str(teamB)]




sumList = [0] * 4915
for i, elem in enumerate(cList):
	sumList[i] = viewsData.ix[i, elem]

print(sumList)




standardData.to_csv('testing.csv')



'''
homeList = list(standardData['Home'])
awayList = list(standardData['Away'])
cList = []

for i, elem in enumerate(homeList):
	cList.append(str(homeList[i])+ ',' + str(awayList[i]))

print(cList)

sumList = [0] * 4915
print(cList[2540])



for i, elem in enumerate(cList):
	sumList[i] = webMetricData.ix[i, elem]

print(cList[2540])



cList2 = []
for i, elem in enumerate(homeList):
	cList2.append(str(awayList[i]) + ',' + str(homeList[i]))

for i, elem in enumerate(cList2):
	try:
		sumList[i] = webMetricData.ix[i, elem]
	except:
		continue


print(sumList)

print(standardData.head())




#print(standardData.head())
sumArr = [0] * 4915

for i, row in enumerate(standardData['Home']):
	sumArr[i] = sumArr[i] + standardData[str(row)]


for i, row in enumerate(standardData['Away']):
	sumArr[i] = sumArr[i] + standardData[str(row)]

standardData['Times_Website_Visited'] = sumArr

print(standardData.describe())

standardData.to_csv('testing.csv')


#print(gameData.head())
#print(standardData.head())
'''

'''
webM_f['sort'] = pd.to_datetime(webM_f['Date'])
webM_f = webM_f.sort_values(by='sort', ascending=True)
webM_f = webM_f.drop('sort', 1)
'''