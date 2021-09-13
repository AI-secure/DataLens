import numpy as np
import argparse

x = [2, 2, 5, 8, 8, 9, 10, 11, 11, 12, 15, 28, 32, 42, 46, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101, 101]
y = [[-1, 2], [-1, 2], [-1, 5], [-1, 8], [-1, 8], [-1, 9], [-1, 10], [-1, 11], [-1, 11], [-1, 12], [-1, 15], [-1, 28], [-1, 32], [-1, 42], [-1, 46], [1451516400, 1483225200], [1451602843, 1483311589], [1451602962, 1483311743], [1451603006, 1483311791], [1451603122, 1483311883], [1451603285, 1483311990], [1451604148, 1483312891], [1451604963, 1483314738], [-122.513648358854, -122.332574620522], [1, 5], [1451603253, 1483316817], [1, 83], [37.6168823239251, 37.8544643401172], [955490400, 1539122400]]
for i in range(1,29):
	x[i] += x[i - 1]

FEATURES = [
	'ALS Unit',
	'Final Priority',
	'Call Type Group',
	'Original Priority',
	'Priority',
	'City',
	'Unit Type',
	'Fire Prevention District',
	'Battalion', 
	'Supervisor District',
	'Call Final Disposition',
	'Zipcode of Incident',
	'Call Type',
	'Neighborhooods - Analysis Boundaries',
	'Station Area',
	'Watch Date',
	'Received DtTm',
	'Entry DtTm',
	'Dispatch DtTm',
	'Response DtTm',
	'On Scene DtTm',
	'Transport DtTm',
	'Hospital DtTm',
	'Location - Lng',
	'Number of Alarms',
	'Available DtTm',
	'Unit sequence in call dispatch',
	'Location - Lat',
	'Call Date',
	'Unit ID',
	'Box',
	'Address',
]

def data2str(ans, n_dim=29):
	temp = ""
	for i in range(n_dim):
		if (i == 0):
			tmp = ans[:x[i]]
		else:
			tmp = ans[x[i - 1]:x[i]]
		_ = np.argmax(tmp)
		if (i == 0):
			temp += str(_)
		else:
			if (x[i] - x[i - 1] == 101):
				if (_ == 100):
					temp += ","
				else:
					step = float(y[i][1] - y[i][0]) / 100
					value = y[i][0] + (_ + 0.5) * step
					if (i != 23 and i != 27):
						temp += "," + str(int(round(value)))
					else:

						temp += "," + str(value)
			else:
				temp += "," + str(_)
	return temp

def batch2str(data, out_file, n_dim=29, n_features=20):
	g = open(out_file, "w+")
	temp = ''
	for i in range(n_features):
		if i > 0:
			temp += ','
		temp += FEATURES[i]
	g.write(temp + "\n")

	for i in range(data.shape[0]):
		temp = data2str(data[i,:], n_dim = n_dim)
		g.write(temp + "\n")
	g.close()

