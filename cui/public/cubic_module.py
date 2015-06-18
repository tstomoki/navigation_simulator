# -*- coding:utf-8 -*-
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import pdb
from scipy import stats

def draw_3d_scatter(xlist, ylist, zlist, x_label, y_label, z_label, column_names, row_names, ylim=None, zlim=None, alpha=None):
	fig   = plt.figure()
	ax    = Axes3D(fig)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
        ax.set_zlabel(z_label)

        # set ylim if defined
        if not ylim is None:
                ax.set_ylim(ylim)
        # set zlim if defined
        if not zlim is None:
                ax.set_zlim(zlim)
        alpha = alpha if not alpha is None else 1.0
        X = np.array(xlist)
        Y = np.array(ylist)
        Z = np.array(zlist)
        ax.scatter3D(np.ravel(X),np.ravel(Y),np.ravel(Z))
        return

def draw_3d_bar(xlist, ylist, zlist, x_label, y_label, z_label, column_names, row_names, ylim=None, zlim=None, alpha=None):
        fig = plt.figure()
	ax = Axes3D(fig)
	
	lx= len(column_names)            # Work out matrix dimensions
	ly= len(row_names)

	xpos = np.arange(0,lx,1)    # Set up a mesh of positions
	ypos = np.arange(0,ly,1)
	xpos, ypos = np.meshgrid(xpos+0.25, ypos+0.25)

	xpos = xpos.flatten()   # Convert positions to 1D array
	ypos = ypos.flatten()
	zpos = np.zeros(lx*ly)
	
	dx = 0.5 * np.ones_like(zpos)
	dy = dx.copy()
        dz = zlist.astype(float).flatten()

        # set ylim if defined
        if not ylim is None:
                ax.set_ylim(ylim)
        # set zlim if defined
        if not zlim is None:
                dz = np.array( [ max(0, num - zlim[0]) for num in dz] )
                zpos  += zlim[0]
                ax.set_zlim(zlim)
        alpha = alpha if not alpha is None else 1.0
                
	ax.bar3d(xpos,ypos,zpos, dx, dy, dz, color='b', alpha=alpha)
	
	#ax.w_xaxis.set_ticklabels(column_names)
	#ax.w_yaxis.set_ticklabels(row_names, rotation=20)
	ax.set_xlabel(x_label, fontsize='small')
	ax.set_ylabel(y_label, fontsize='small')
	ax.set_zlabel(z_label, fontsize='small')
	return 

# 時間(X)ごとのラインを表示するメソッド(X:時間(unitTime), Y:LogSpeed, Z:馬力)
def showLineFrame3DGraph(title, output_path, coefficientList, yList = [10,11,12,13,14,15,16,17,18,19,20], yMesh = False):

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_xlabel("Time")
	ax.set_ylabel("Log Speed [knot]")
        z_label = "$\Delta$" + "SHP [kW]"
	ax.set_zlabel(z_label)

	xPList = []
	yPList = []
	zPList = []

	for co in coefficientList:
		xList = []
		zList = []
		for y in yList:
			xList.append(co[0])
			z = co[1]*(y**3) + co[2]*(y**2) + co[3]*(y) + co[4]
			zList.append(z)

		X = np.array(xList)
		Y = np.array(yList)
		Z = np.array(zList)

		ax.plot_wireframe(np.ravel(X),np.ravel(Y),np.ravel(Z))

		xPList.append(xList)
		yPList.append(yList)
		zPList.append(zList)


	if yMesh:
		for i in range(len(yList)):
			_xList = []
			_yList = []
			_zList = []
			
			for j in range(len(yPList)):
				_xList.append(xPList[j][i])
				_yList.append(yList[i])
				_zList.append(zPList[j][i])

			X = np.array(_xList)
			Y = np.array(_yList)
			Z = np.array(_zList)
			ax.plot_wireframe(np.ravel(X),np.ravel(Y),np.ravel(Z))
			

	plt.show();

# 時間(X)ごとのラインを表示するメソッド(X:時間(unitTime), Y:LogSpeed, Z:馬力)
def saveLineFrame3DGraph(title, output_path, coefficientList, yList = [10,11,12,13,14,15,16,17,18,19,20], yMesh = False):
        #graphクリア
        plt.clf()
#        import pdb; pdb.set_trace()
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_xlabel("Time")
	ax.set_ylabel("Log Speed [knot]")
        z_label = "$\Delta$" + "SHP [kW]"
	ax.set_zlabel(z_label)

	xPList = []
	yPList = []
	zPList = []

	for co in coefficientList:
		xList = []
		zList = []
		for y in yList:
			xList.append(co[0])
			z = co[1]*(y**3) + co[2]*(y**2) + co[3]*(y) + co[4]
			zList.append(z)

		X = np.array(xList)
		Y = np.array(yList)
		Z = np.array(zList)
		ax.plot_wireframe(np.ravel(X),np.ravel(Y),np.ravel(Z))

		xPList.append(xList)
		yPList.append(yList)
		zPList.append(zList)


	if yMesh:
		for i in range(len(yList)):
			_xList = []
			_yList = []
			_zList = []
			
			for j in range(len(yPList)):
				_xList.append(xPList[j][i])
				_yList.append(yList[i])
				_zList.append(zPList[j][i])

			X = np.array(_xList)
			Y = np.array(_yList)
			Z = np.array(_zList)
			ax.plot_wireframe(np.ravel(X),np.ravel(Y),np.ravel(Z))

        x_label = [ co[8].strftime('%y-%m') for co in coefficientList]
        temp_indexes = []
        xPList = reduce(lambda x,y: x+y,xPList)
        for _list_element in xPList:
                if _list_element not in temp_indexes:
                        temp_indexes.append(_list_element)
        #        plt.rc('xtick', labelsize=10)
        
        #        temp_indexes = [ index -1 for index in temp_indexes]
        temp_indexes, x_label = makeIntervalLabel(temp_indexes, x_label)
        plt.xticks(temp_indexes, x_label, rotation=30)
        title = title + '\n'
        plt.title(title, fontweight="bold")
        #import pdb; pdb.set_trace()
        plt.savefig(output_path)
        return

# 散布図で表示するメソッド(X:時間(unitTime), Y:LogSpeed, Z:馬力)
def showScatter3DGraph(title, output_path, coefficientList, yList = [10,11,12,13,14,15,16,17,18,19,20]):

	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_xlabel("Time")
	ax.set_ylabel("Log Speed [knot]")
        z_label = "$\Delta$" + "SHP [kW]"
	ax.set_zlabel(z_label)

	for co in coefficientList:
		xList = []
		zList = []
		for y in yList:
			xList.append(co[0])
			z = co[1]*(y**3) + co[2]*(y**2) + co[3]*(y) + co[4]
			zList.append(z)

		X = np.array(xList)
		Y = np.array(yList)
		Z = np.array(zList)

		ax.scatter3D(np.ravel(X),np.ravel(Y),np.ravel(Z))

        x_label = [ co[8].strftime('%y-%m') for co in coefficientList]
        temp_indexes = []
        xPList = reduce(lambda x,y: x+y,xPList)
        for _list_element in xPList:
                if _list_element not in temp_indexes:
                        temp_indexes.append(_list_element)
        #        plt.rc('xtick', labelsize=10)
        #        temp_indexes = [ index -1 for index in temp_indexes]
        temp_indexes, x_label = makeIntervalLabel(temp_indexes, x_label)
        plt.xticks(temp_indexes, x_label, rotation=30)
        title = title + '\n'
        plt.title(title, fontweight="bold")

	plt.show()
        return

# 散布図で表示するメソッド(X:時間(unitTime), Y:LogSpeed, Z:馬力)
def saveScatter3DGraph(title, output_path, coefficientList, yList = [10,11,12,13,14,15,16,17,18,19,20]):
        #graphクリア
        plt.clf()
	fig = plt.figure()
	ax = Axes3D(fig)
	ax.set_xlabel("Time")
	ax.set_ylabel("Log Speed [knot]")
        z_label = "$\Delta$" + "SHP [kW]"
	ax.set_zlabel(z_label)
        temp_indexes = []

	for co in coefficientList:
		xList = []
		zList = []
		for y in yList:
			xList.append(co[0])
			z = co[1]*(y**3) + co[2]*(y**2) + co[3]*(y) + co[4]
			zList.append(z)

		X = np.array(xList)
		Y = np.array(yList)
		Z = np.array(zList)

		ax.scatter3D(np.ravel(X),np.ravel(Y),np.ravel(Z))
                temp_indexes.append(co[0])

        x_label = [ co[8].strftime('%y-%m') for co in coefficientList]
        #        temp_indexes = [ index -1 for index in temp_indexes]
        #        plt.rc('xtick', labelsize=10)
        temp_indexes, x_label = makeIntervalLabel(temp_indexes, x_label)
        plt.xticks(temp_indexes, x_label, rotation=30)
        title = title + '\n'
        plt.title(title, fontweight="bold")
        plt.savefig(output_path)
        
        return

def saveEachSpeed(output_dir_path,coefficientList, yList):
        #graphクリア
        plt.clf()
        for speed in yList:
                time = []
                shp = []
                for co in coefficientList:
			time.append(co[0])
			temp_shp = co[1]*(speed**3) + co[2]*(speed**2) + co[3]*(speed) + co[4]
			shp.append(temp_shp)
                time = np.array(time)
                shp = np.array(shp)
                
                #グラフsave
                title = 'error_func_at' + str(speed) + 'knot'
                z_label = "$\Delta$" + "SHP [kW]"
                graphInitializer(title, "Time", z_label)
                plt.xlim([datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2014, 1, 1, 0, 0)])
                plt.xticks(rotation=30)
                plt.scatter(time, shp,facecolor='b', edgecolor='b',s=100)
                output_path = output_dir_path + '/' + title + '.png'
                outputByMode(1, output_path)

                #グラフsave(近似式付き)
                title = 'error_func_at' + str(speed) + 'knot_with_approx'
                z_label = "$\Delta$" + "SHP [kW]"
                graphInitializer(title, "Time", z_label)
                plt.xlim([datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2014, 1, 1, 0, 0)])
                plt.xticks(rotation=30)
                plt.scatter(time, shp,facecolor='b', edgecolor='b',s=100)

                #近似式の描画
                #datetimeからエポック時間に変換
                x = datetime2epoch(time)
                slope, intercept, r_value, _, _ = stats.linregress(x, shp)
                approx_opt_y = calclinearFunc(x, slope, intercept)
                #approx_x, approx_opt_y = estimate_module(x, y, 1)
                #エポック時間からdatetimeへ変換
                approx_x = epoch2datetime(x)
                plt.plot(approx_x, approx_opt_y, color='r')
                output_path = output_dir_path + '/' + title + '.png'
                outputByMode(1, output_path)

def saveEachSpeedPercentage(output_dir_path,coefficientList, yList, unique_keys, item):
        #graphクリア
        plt.clf()
        for speed in yList:
                time = []
                shp = []
                for i, co in enumerate(coefficientList):
			time.append(co[0])
			temp_shp = co[1]*(speed**3) + co[2]*(speed**2) + co[3]*(speed) + co[4]
                        average_shp = getAverageSHPatDesignatedSpan(unique_keys[i], item)
			shp.append(temp_shp)
                time = np.array(time)
                shp = np.array(shp)
                shp_percentage = shp/average_shp * 100
                
                #グラフsave
                title = 'error_func_at' + str(speed) + 'knot'
                z_label = "$\Delta$" + "SHP [%]"
                graphInitializer(title, "Time", z_label)
                plt.xlim([datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2014, 1, 1, 0, 0)])
                plt.xticks(rotation=30)
                plt.scatter(time, shp_percentage,facecolor='b', edgecolor='b',s=100)
                output_path = output_dir_path + '/' + title + '.png'
                outputByMode(1, output_path)

                #グラフsave(近似式付き)
                title = 'error_func_at' + str(speed) + 'knot_with_approx'
                z_label = "$\Delta$" + "SHP [%]"
                graphInitializer(title, "Time", z_label)
                plt.xlim([datetime.datetime(2011, 1, 1, 0, 0), datetime.datetime(2014, 1, 1, 0, 0)])
                plt.xticks(rotation=30)
                plt.scatter(time, shp_percentage,facecolor='b', edgecolor='b',s=100)

                #近似式の描画
                #datetimeからエポック時間に変換
                x = datetime2epoch(time)
                slope, intercept, r_value, _, _ = stats.linregress(x, shp_percentage)
                approx_opt_y = calclinearFunc(x, slope, intercept)
                #approx_x, approx_opt_y = estimate_module(x, y, 1)
                #エポック時間からdatetimeへ変換
                approx_x = epoch2datetime(x)
                plt.plot(approx_x, approx_opt_y, color='r')
                output_path = output_dir_path + '/' + title + '.png'
                outputByMode(1, output_path)

#coefficientList = [[0,0.001,-0.01,0.1,0],[1,-0.001,0.01,0.1,1],[2,0.001,0.01,0.1,2]]
#showLineFrame3DGraph(coefficientList, yMesh = True)
