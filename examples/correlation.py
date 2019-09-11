import pyinsights.dataread as dr


def correlation_example():
	data = dr.data_read(fname='candy-data.csv', norm=['winpercent'])
	dr.data_plot(data=data)

def apriori_example():
	data = dr.data_read(fname='candy-data.csv', norm=['winpercent'])
	dr.data_apri(data=data, keyel='winpercent')


if __name__ == '__main__':
	from matplotlib.pyplot import show
	correlation_example()
	apriori_example()
	show()


