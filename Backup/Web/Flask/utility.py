import random
import pandas as pd
import geopandas as gpd
from os import path
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import LineString
import numpy as np
import osmnx as ox
import plotly.graph_objects as go 
from sklearn.ensemble import IsolationForest
from shapely.geometry import Point, mapping
from functools import partial
import pyproj
from shapely.ops import transform
from scipy.spatial.distance import cdist
import matplotlib
import shapely
matplotlib.use('Agg')


class utility:
    def __init__(self,staticFold):
        self.loadDatasetItaly(staticFold)
        self.loadDatasetMilan(staticFold)

    def loadDatasetMilan(self,staticFold):
        #Load shapefile Municipi
        self.municipi = gpd.read_file(path.join(staticFold, 'Dataset/Municipi/Municipi.shp')).sort_values(by=['MUNICIPIO']).reset_index().drop(["index"],axis = 1).to_crs(epsg=3857)
        #Load shapefile NIL
        nildf =  pd.read_csv(path.join(staticFold, 'Dataset/nilComplete.csv'))
        self.nil = gpd.GeoDataFrame(
            nildf.loc[:, [c for c in nildf.columns if c != "geometry"]],
            geometry=gpd.GeoSeries.from_wkt(nildf["geometry"]))
        self.nil = self.nil.set_crs('epsg:4326')
        self.nil = self.nil.to_crs(epsg=3857)
        #Load dataset population
        self.popMilano = pd.read_csv(path.join(staticFold, 'Dataset/popMilanComplete.csv'))
        #Load prices
        self.prices =  gpd.read_file(path.join(staticFold, 'Dataset/pricesTot.csv'),GEOM_POSSIBLE_NAMES="geometry",KEEP_GEOM_COLUMNS="NO")
        #Load shapefile CAP
        self.capZone = gpd.read_file(path.join(staticFold, 'Dataset/banchedati-cap-zone-demo-database')).sort_values(by=['CAP']).reset_index().drop(["index"],axis = 1).to_crs(epsg=3857)
        #Load incomes
        self.redditi = pd.read_csv(path.join(staticFold, "Dataset/incomes.csv"))
        #Movements
        #Using matplotlib (only png, non interactive)
        self.movShap = gpd.read_file(path.join(staticFold, "Dataset/ShapeMatrice OD2016 - Passeggeri - Zone interne"))
        self.movShapWM = self.movShap.to_crs(epsg=3857)
        self.mov = pd.read_csv(path.join(staticFold, "Dataset/movementsToMilan.csv"))
        self.latLong = pd.read_csv(path.join(staticFold, "Dataset/italy_lng_latAgg2021.csv"))
        self.prov = pd.read_csv(path.join(staticFold, "Dataset/provincia.csv"),sep = ";")
        ox.settings.use_cache=True

    def loadDatasetItaly(self,staticFold):
        self.df= pd.read_csv(path.join(staticFold, "Dataset/clusteringItaliaSOM95SS.csv"))
        clusterNames = []
        for i in range(0,max(self.df.cluster)+1):#For each cluster
            if not self.df.loc[self.df["cluster"] == i].empty:
                clusterName = ""
                for col in self.df.columns: #For each "important" column
                    if (col not in ["Value","Territorio","lat","lng","Size km2","populPerKm2","cluster","ITTER107"]):
                        if col == "Redditi 2020":
                            clusterName += str(int(self.df.loc[self.df["cluster"] == i][col].mean())) + " "
                        elif col == "AvgNumberOfComponents":
                            clusterName += str(round(self.df.loc[self.df["cluster"] == i][col].mean(),2)) + " "
                        else:
                            clusterName += str(self.df.loc[self.df["cluster"] == i][col].value_counts().idxmax()) + " "
                clusterNames.append(clusterName + str(i))

        splitted_search=[x.split(" ") for x in clusterNames]
        self.clusterTypes = pd.DataFrame(splitted_search, columns =['Gender', 'MaritalStatus', 'AgeGroup',
                                                            'AvgIncome','AvgNumComp','cluster'], dtype = str)
        self.clusterTypes["AvgIncome"] = self.clusterTypes["AvgIncome"].astype(int)
        self.clusterTypes["cluster"] = self.clusterTypes["cluster"].astype(int)
        #Load shapefile italy for plotting
        self.italyShap = gpd.read_file(path.join(staticFold, "Dataset/Italy/Shapefile")).reset_index().drop(["index"],axis = 1).to_crs(epsg=4326)
        #Load csv italy for plotting
        self.italy = pd.read_csv(path.join(staticFold, "Dataset/italy.csv"))


    def kmRadius(self,c): #Define area of attractiveness
        point = Point(c)
        
        local_azimuthal_projection = f"+proj=aeqd +R=6371000 +units=m +lat_0={point.y} +lon_0={point.x}"

        wgs84_to_aeqd = partial(
            pyproj.transform,
            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
            pyproj.Proj(local_azimuthal_projection),
        )

        aeqd_to_wgs84 = partial(
            pyproj.transform,
            pyproj.Proj(local_azimuthal_projection),
            pyproj.Proj('+proj=longlat +datum=WGS84 +no_defs'),
        )

        point_transformed = transform(wgs84_to_aeqd, point)

        buffer = point_transformed.buffer(20_000)#Meters

        buffer_wgs84 = transform(aeqd_to_wgs84, buffer)
        dic = mapping(buffer_wgs84)
        return np.asarray(dic['coordinates'])[0]

    ##Mean based on: municipalities with more target population per sqmt influence more (can be **2 or **3)
    ##Remove far away points from calculation
    def calculateWeightedMean(self,lat, lon, populationsqm, clf):
        avLat = np.average(a=lat[clf > 0], weights=populationsqm[clf > 0]**3)
        avLon = np.average(a=lon[clf > 0], weights=populationsqm[clf > 0]**3)
        return avLon,avLat

    def closest_point(self,point, points):
        """ Find closest point from a list of points. """
        return points[cdist([point], points).argmin()]

    def getItalyMap(self,ageGroup,gender,maritalStatus,income,uid):
        currClusters = self.clusterTypes.loc[(self.clusterTypes["AgeGroup"] == ageGroup) & 
                                        (self.clusterTypes["Gender"] == gender) & 
                                        (self.clusterTypes["MaritalStatus"] == maritalStatus)]
        def takeClosest(curr, targetIncome,alreadySelected):
            closestIncome = curr.iloc[:1]["AvgIncome"].values[0]
            for i in range(1, curr.shape[0]):    
                currInc = curr.iloc[i:i+1]["AvgIncome"].values[0]
                if (abs(currInc - targetIncome) < abs(closestIncome - targetIncome)) and (currInc not in alreadySelected):
                    closestIncome = currInc
            return closestIncome
        if currClusters.shape[0]!=0:
            #Select points
            regions = []
            cities = []
            clusters = []
            alreadySelectedInc = []
            for _ in range (0,5):#Get top 5 regions with
                inc = takeClosest(currClusters,income,alreadySelectedInc)
                curr = currClusters.loc[currClusters["AvgIncome"] == inc]#Text of current cluster
                if inc not in alreadySelectedInc:#If it is already done
                    alreadySelectedInc.append(inc)
                    clusters.append(int(curr["cluster"].values[0]))
                    selectedClusterPoints = self.df.loc[self.df["cluster"] == curr["cluster"].values[0]]

                    clf = IsolationForest( random_state=0).fit_predict(selectedClusterPoints[["lng","lat"]].values)#Schema a blochi pptx

                    lonC, latC = self.calculateWeightedMean(selectedClusterPoints["lat"], 
                                selectedClusterPoints["lng"], selectedClusterPoints["populPerKm2"],clf)
                    closest = self.closest_point((lonC,latC), list(zip(selectedClusterPoints["lng"], selectedClusterPoints["lat"])))
                    cities.append(selectedClusterPoints.loc[(selectedClusterPoints["lng"] == closest[0])&
                                            (selectedClusterPoints["lat"] == closest[1])]["Territorio"].values[0])#Get city of center of cluster
                                
                    point = gpd.points_from_xy([closest[0]], [closest[1]]) 
                    regions.append(self.italyShap[self.italyShap.contains(point[0])]["DEN_REG"].values[0])#Get region of center of cluster

            _, ax = plt.subplots(figsize=(15, 15))
            self.italyShap["where"]=np.zeros(len(self.italyShap))
            self.italyShap.loc[self.italyShap['DEN_REG'].isin(regions),'where'] = 10.0
            self.italyShap.plot(column= self.italyShap["where"], alpha=0.5, edgecolor='k',ax=ax,cmap = "Reds")
            plt.axis('off')
            ax.figure.suptitle("Regioni con cluster selezionato")
            img = "static/Maps/"+uid+"/italyMap.png"
            ax.figure.savefig(img)
            retList = []
            retList.append(img+"?rand=" + str(random.randint(0, 1000)))
            retList.append(cities)
            retList.append(regions)
            retList.append(clusters)
            return retList
        else:
            return
    
    def plotItalyCluster(self,clusterNumber,region,uid):
        selectedClusterPoints = self.df.loc[self.df["cluster"] == clusterNumber]
        curr = self.clusterTypes.loc[self.clusterTypes["cluster"] == clusterNumber]
        if selectedClusterPoints.shape[0]!=0:
            clf = IsolationForest( random_state=0).fit_predict(selectedClusterPoints[["lng","lat"]].values)#Schema a blochi pptx

            lonC, latC = self.calculateWeightedMean(selectedClusterPoints["lat"], 
                        selectedClusterPoints["lng"], selectedClusterPoints["populPerKm2"],clf)
            circle = self.kmRadius((lonC,latC))
            closest = self.closest_point((lonC,latC), list(zip(selectedClusterPoints["lng"], selectedClusterPoints["lat"])))#Get closest point from picked

            clf = np.where(clf > 0, "#BBF90F", "#929591")            
            scatt = []

            color = np.append(np.repeat("#0000FF",len(circle[:,1])),"#FF4500")         
            xp = np.append(circle[:,0],lonC)
            yp = np.append(circle[:,1],latC)
            zp = np.append(np.zeros(len(circle[:,1])),0)
            # Circle + center
            scatt.append(go.Scatter3d(x=xp, y=yp, z=zp,showlegend=False,
                                        mode='markers',
                                        hoverinfo='skip',
                                        #+ '%{customdata[2]}' + '%{customdata[3]}',
                                        marker=dict(
                                                color=color,                # set color to an array/list of desired values
                                                colorscale='Viridis',   # choose a colorscale
                                                opacity=0.8,
                                            )
                                        )
                        )
            
            #Add data cluster
            customData = selectedClusterPoints[["Territorio","populPerKm2"]]#selectedClusterPoints[["Territorio","Sesso","Stato civile","AgeGroup"]],
            customData["populPerKm2"] = customData["populPerKm2"].astype(int)
            # customData.loc[len(customData)] = ["area of", "attractiveness"]
            # print(customData)
            xp = selectedClusterPoints["lng"].values
            yp = selectedClusterPoints["lat"].values
            zp = selectedClusterPoints["populPerKm2"].values
            #Points
            scatt.append(go.Scatter3d(x=xp, y=yp, z=zp,
                                        showlegend=False,
                                        mode='markers',
                                        customdata=customData,
                                        hovertemplate='%{customdata[0]}'+' \n Popolazione per km2: %{customdata[1]} ',
                                        #+ '%{customdata[2]}' + '%{customdata[3]}',
                                        marker=dict(
                                                color=clf,                # set color to an array/list of desired values
                                                colorscale='Viridis',   # choose a colorscale
                                                opacity=0.8,
                                            )
                                        )
                        )            
            #Add vertical lines
            linex = np.array(list(zip(xp,xp,np.full(len(xp), None)))).flatten()
            liney = np.array(list(zip(yp,yp,np.full(len(xp), None)))).flatten()
            linez = np.array(list(zip(zp,np.zeros(len(xp)),np.full(len(xp), None)))).flatten()
            scatt.append(go.Scatter3d(x=linex, y=liney, z=linez,
                                        showlegend=False,
                                        mode='lines',hoverinfo='skip',
                                        marker=dict(
                                            color="black",                # set color to an array/list of desired values
                                            colorscale='Viridis',   # choose a colorscale
                                            opacity=0.8,
                                        )
                                        ))
            
            for i in range (0,len(self.italyShap)):
                if self.italyShap ["DEN_REG"][i] == region:
                    if isinstance(self.italyShap["geometry"][i], shapely.geometry.multipolygon.MultiPolygon):
                        for polygon in self.italyShap["geometry"][i]:
                            x = np.asarray(polygon.exterior.coords.xy[0])
                            y = np.asarray(polygon.exterior.coords.xy[1])
                            z = np.zeros(len(polygon.exterior.coords.xy[1]))
                            scatt.append(go.Scatter3d(x=x, y=y, z=z,showlegend=False,
                                                mode='lines',hoverinfo='skip',
                                                marker=dict(
                                                    color="lime",                # set color to an array/list of desired values
                                                    colorscale='Viridis',   # choose a colorscale
                                                    opacity=0.8,
                                                )
                                                ))
                    else:
                        x = np.asarray(self.italyShap["geometry"][i].exterior.coords.xy[0])
                        y = np.asarray(self.italyShap["geometry"][i].exterior.coords.xy[1])
                        z = np.zeros(len(self.italyShap["geometry"][i].exterior.coords.xy[1]))

                        scatt.append(go.Scatter3d(x=x, y=y, z=z,showlegend=False,
                                                mode='lines',hoverinfo='skip',
                                                marker=dict(
                                                    color="lime",                # set color to an array/list of desired values
                                                    colorscale='Viridis',   # choose a colorscale
                                                    opacity=0.8,
                                                )
                                                ))
            text = str(curr.values[0][0]) + " " + str(curr.values[0][1]) + " " + str(curr.values[0][2]) + " Redditi: " + str(curr.values[0][3]) + " Numero medio di componenti: " + str(curr.values[0][4])

            text += '<br>' + 'Center is located in: %s' % (selectedClusterPoints.loc[(selectedClusterPoints["lng"] == closest[0])&
                                        (selectedClusterPoints["lat"] == closest[1])]["Territorio"].values[0])
            camera = dict(
                eye=dict(x=0, y=0, z=1),
                up=dict(x=0, y=1, z=0)
            )
            fig = go.Figure(data=scatt)
            fig.update_layout(scene = dict(
                    zaxis_title='Popolazione per km2',
                    xaxis_title='',
                    yaxis_title='',
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False)
                    ))
            fig.update_layout(scene_camera=camera)

            url = "static/Maps/"+uid+"/clusterItaly.html"
            fig.write_html(url)
            return [text,url]
        else:
            return #No clusters found
        #Careful with -1 if use HDBSCAN

    def plotMunicipi(self,ageGroup,gender,maritalStatus,numComp,uid):
        considered = self.popMilano.loc[(self.popMilano["Classe_eta_capofamiglia"] == ageGroup)&
                        (self.popMilano["Genere_capofamiglia"] == gender)&
                        (self.popMilano["Tipologia_familiare"] == maritalStatus)&
                        (self.popMilano["Numero_componenti"] == numComp)]
        considered = considered.sort_values(by=['Municipio']).reset_index().drop(["index"],axis = 1)# Sort 
        if len(considered) != 9 or considered.empty: #Need to have data for all Municipi
            return
        _, ax = plt.subplots(figsize=(15, 15))
        color = "Reds"
        vmin = considered["Frq"].min()
        vmax = considered["Frq"].max()
        step = (vmax-vmin)/9
        self.municipi.plot(column= considered["Frq"], alpha=0.5, edgecolor='k', vmin = vmin, 
                    vmax = vmax,cmap =color, ax=ax, legend=True, 
                    legend_kwds={'label': 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp),
                                'orientation': "horizontal","location": "top","shrink":0.87,
                                "ticks": np.arange(vmin,vmax+step,step).astype(int)})
        cx.add_basemap(ax,source="static/Maps/MilanSmall.tif")
        plt.axis('off')
        self.municipi.apply(lambda x: ax.annotate(text=x['MUNICIPIO'], xy=x.geometry.centroid.coords[0], ha='center',fontsize=20), axis=1)
        img = "static/Maps/"+uid+"/municipiMilan.png"
        ax.figure.savefig(img)
        retList = []
        retList.append(img+"?rand=" + str(random.randint(0, 1000)))
        retList.append(considered.sort_values(by=['Frq'],ascending = False).iloc[0:3]["Municipio"].values.tolist())
        retList.append(considered.sort_values(by=['Frq'],ascending = False).iloc[0:3]["Frq"].values.tolist())
        return retList

    def plotNIL(self,ageGroup,gender,maritalStatus,numComp,uid):
        considered = self.popMilano.loc[(self.popMilano["Classe_eta_capofamiglia"] == ageGroup)&
                        (self.popMilano["Genere_capofamiglia"] == gender)&
                        (self.popMilano["Tipologia_familiare"] == maritalStatus)&
                        (self.popMilano["Numero_componenti"] == numComp)]
        considered = considered.sort_values(by=['Municipio']).reset_index().drop(["index"],axis = 1)# Sort 

        consideredNIL = pd.merge(self.nil,considered,how='inner',left_on=['Municipio'],right_on=['Municipio'],suffixes=('', '_y'))
        consideredNIL["RelFrq"] = (consideredNIL["Frq"]*consideredNIL["Popolazione"]/consideredNIL["PopolazioneTot"]).astype(int)
        if consideredNIL.empty:
            return
        if len(consideredNIL["RelFrq"]) == len(self.nil):
            vmin = consideredNIL["RelFrq"].min()
            _, ax = plt.subplots(figsize=(15, 15))
            color = "Reds"
            vmax = consideredNIL["RelFrq"].max()
            step = (vmax-vmin)/9
            consideredNIL.plot(column= consideredNIL["RelFrq"], alpha=0.5, edgecolor='k', vmin = vmin, 
                        vmax = vmax,cmap =color, ax=ax, legend=True, 
                        legend_kwds={'label': 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp),
                                    'orientation': "horizontal","location": "top","shrink":0.87,
                                    "ticks": np.arange(vmin,vmax+step,step).astype(int)})
            cx.add_basemap(ax,source="static/Maps/MilanSmall.tif")
            plt.axis('off')
            consideredNIL.apply(lambda x: ax.annotate(text=x['NIL'], xy=x.geometry.centroid.coords[0], ha='center',fontsize=7), axis=1)
            img = "static/Maps/"+uid+"/NILMilan.png"
            ax.figure.savefig(img)
            retList = []
            retList.append(img+"?rand=" + str(random.randint(0, 1000)))
            retList.append(consideredNIL.sort_values(by=['RelFrq'],ascending = False).iloc[0:3]["NIL"].values.tolist())
            retList.append(consideredNIL.sort_values(by=['RelFrq'],ascending = False).iloc[0:3]["RelFrq"].values.astype(int).tolist())
            return retList
        else:
            return 

    def plotIncomes(self,income,uid):
        consideredRed = self.redditi.loc[(self.redditi["Redditi e variabili IRPEF"] == income)]
        if consideredRed.empty:
            return
        consideredRed = consideredRed.sort_values(by=['CAP']).reset_index().drop(["index"],axis = 1)# Sort 

        vmin = consideredRed["Valori"].min()
        _, ax = plt.subplots(figsize=(15, 15))
        color = "Reds"
        vmax = consideredRed["Valori"].max()
        step = (vmax-vmin)/9
        self.capZone.plot(column= consideredRed["Valori"], alpha=0.5, edgecolor='k', vmin = vmin, 
                vmax = vmax,cmap =color, ax=ax, legend=True, 
                legend_kwds={'label': income,
                                'orientation': "horizontal","location": "top","shrink":0.87,
                                "ticks": np.arange(vmin,vmax+step,step).astype(int)})
        cx.add_basemap(ax,source="static/Maps/MilanSmall.tif")
        plt.axis('off')
        self.capZone.apply(lambda x: ax.annotate(text=x['CAP'], xy=x.geometry.centroid.coords[0], ha='center',fontsize=20), axis=1)
        img = "static/Maps/"+uid+"/incomeSelected.png"
        ax.figure.savefig(img)
        retList = []
        retList.append(img+"?rand=" + str(random.randint(0, 1000)))
        retList.append(consideredRed.sort_values(by=['Valori'],ascending = False).iloc[0:3]["CAP"].values.tolist())
        retList.append(consideredRed.sort_values(by=['Valori'],ascending = False).iloc[0:3]["Valori"].values.tolist())
        return retList

    
    def plotStores(self,position,storeType,storeName,storeBrand,uid):
        selectedZone = self.movShap.loc[self.movShap["desc_zona"] == position]
        typeOfShop = ox.geometries.geometries_from_polygon(selectedZone["geometry"].values[0], tags = { 'shop':storeType})
        west, south, east, north = selectedZone.total_bounds
        points = gpd.points_from_xy([west,east], [south,north])
        print(west, south, east, north)
        typeOfShop = typeOfShop.to_crs(epsg=3857)
        if(len(typeOfShop) == 0):
            return
        ax = typeOfShop.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', color = "b")
        print(points)
        pointsGdf = gpd.GeoDataFrame(points,geometry = points)
        pointsGdf= pointsGdf.set_crs('epsg:4326')
        pointsGdf = pointsGdf.to_crs(epsg=3857)
        pointsGdf.plot(ax=ax,alpha = 0)
        cx.add_basemap(ax,source="static/Maps/MilanSmall.tif")
        plt.axis('off')
        if storeName is not None and storeName != '':
            subType = ox.geometries.geometries_from_polygon(selectedZone["geometry"].values[0], 
                            tags = { 'name':[i for i in storeName.split(",") if i]})
            subType = subType.to_crs(epsg=3857)
            subType.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', ax=ax, color = "r")
        if storeBrand is not None and storeBrand != '':
            subType = ox.geometries.geometries_from_polygon(selectedZone["geometry"].values[0], 
                            tags = { 'brand':[i for i in storeBrand.split(",") if i]})
            subType = subType.to_crs(epsg=3857)
            subType.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', ax=ax, color = "g")

        ax.figure.suptitle("Red points: store names, green points: store brands")
        img = "static/Maps/"+uid+"/stores.png"
        ax.figure.savefig(img)
        return img+"?rand=" + str(random.randint(0, 1000))

    def plotMovements(self,position,uid):
        provinceOfTarget = "MI"
        movTarget = self.mov.loc[self.mov["ZONA_DEST"]== position]#Keep Only destination target
        outsideMilan = movTarget.groupby(['ZONA_DEST','PROV_ORIG'], as_index=False).sum()
        insideMilan = movTarget.groupby(['ZONA_DEST','PROV_ORIG','ZONA_ORIG'], as_index=False).sum()#Movement outside Milan
        insideMilan = insideMilan.loc[insideMilan["PROV_ORIG"] ==provinceOfTarget]#Movement inside Milan
        thresh = 500
        images = []
        for movType in ["STU","LAV","OCC","AFF","RIT"]:#Analyze each type individually
            filter_col = [col for col in outsideMilan if col.startswith(movType)]
            currGroupOutside = outsideMilan[outsideMilan[filter_col].sum(axis = 1) > thresh][
                ["ZONA_DEST","PROV_ORIG"] + filter_col]#See if there is at least one type w enough mov
            currGroupOutside = currGroupOutside[currGroupOutside.PROV_ORIG != provinceOfTarget]#Take away inside
            currGroupInside = insideMilan[insideMilan[filter_col].sum(axis = 1) > thresh][
                ["ZONA_DEST",'PROV_ORIG','ZONA_ORIG'] + filter_col]#See if there is at least one type w enough mov
            images.append(self.plotMov(currGroupOutside,currGroupInside,position,movType,filter_col,uid))

        
        return [item for sublist in images for item in sublist]#flat list of images

    def plotMov(self,currGroupOutside,currGroupInside,target, movType,filter_col,uid):
    #Outside movements
        img = []
        
        if currGroupOutside.empty:
            print('No consistent outside traffic to target area')
        else:
            out = []
            for index,row in currGroupOutside[~currGroupOutside["PROV_ORIG"].isin(self.prov["CodSiglaProvincia"])].iterrows():
                out.append("Spostamenti verso " + target + ", tipo: " + movType + " da " +
                                row["PROV_ORIG"] + ": " + str(int(row[filter_col].sum())))
            #Get location of outside points
            currGroupOutside = pd.merge(currGroupOutside,self.prov,how="inner", left_on="PROV_ORIG", right_on="CodSiglaProvincia")#Attach city name from province
            currGroupOutside = pd.merge(currGroupOutside,self.latLong,how="inner", left_on="DescrProvincia", right_on="comune")#Attach lng lat from city name
            #Plot oustide points with lines and write amount
            gdf = gpd.GeoDataFrame(
                    currGroupOutside, geometry=gpd.points_from_xy(currGroupOutside["lng"],currGroupOutside["lat"], crs="EPSG:4326"))
            gdf = gdf.to_crs(epsg=3857)
            gdf = pd.concat([gdf,self.movShapWM.loc[self.movShapWM["desc_zona"] == target]])
            ax = gdf.plot(figsize=(20, 20), alpha=0.5, edgecolor='k',color = "y", markersize = 400)
            cx.add_basemap(ax,source="static/Maps/northernItaly.tif",reset_extent=False)

            plt.axis('off')
            ax.set_title("Spostamenti verso " + target + ", tipo: " + movType + " da fuori provincia")

            #Print non-region data (plot is too big)
            for s in out:
                img.append(s)#Append where are movements from outside (not img just string)

            gdf.apply(lambda x: ax.annotate(#Annotate how many moves
                text=int(x[filter_col].sum()), xy=x.geometry.centroid.coords[0], ha='center',fontsize=20), axis=1)

            #plot lines
            for i in range(0, len(gdf)-1):
                ls = LineString([gdf["geometry"][i], self.movShapWM.loc[self.movShapWM[
                    "desc_zona"] == target].geometry.centroid.item()])
                ax.plot(*ls.xy, color = "green")

            gdf= gdf[gdf["PROV_ORIG"].notna()]
            gdf.apply(lambda x: 
                ax.annotate(#Put name of centres
                    text=x['PROV_ORIG'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]-5000), ha='center',fontsize=15),axis=1)
            outPic = "static/Maps/"+uid+"/outside" + movType +".png"   
            ax.figure.savefig(outPic)
            img.append(outPic+"?rand=" + str(random.randint(0, 1000)))    
                
        #Inside movements
        if currGroupInside.empty or len(currGroupInside) ==1:
            print('No consistent inside traffic to target area')
        else:
            #Get location of inside points
            currGroupInside = pd.merge(self.movShapWM,currGroupInside, how="inner", left_on="desc_zona", right_on="ZONA_ORIG")
            #Plot inside points with lines and write amount
            gdf = gpd.GeoDataFrame(
                    currGroupInside, geometry=currGroupInside.geometry.centroid)
            
            ax = gdf.plot(marker='*',figsize=(20, 20), alpha=0.5, edgecolor='k')
            try:
                cx.add_basemap(ax,source="static/Maps/MilanSmall.tif",reset_extent=False)
            except:
                cx.add_basemap(ax,source="static/Maps/Milan.tif")
            plt.axis('off')
            ax.set_title("Spostamenti verso " + target + ", tipo: " + movType + " da dentro provincia")
            gdf.apply(lambda x: ax.annotate(#Annotate how many moves
                text=int(x[filter_col].sum()), xy=x.geometry.centroid.coords[0], ha='center',fontsize=20)
                    if (x.desc_zona!= "MILANO 2")else None, axis=1)
            gdf.apply(lambda x: ax.annotate(#Annotate how many moves
                text=int(x[filter_col].sum()), xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]-1500), ha='center',fontsize=20)
                    if (x.desc_zona== "MILANO 2")else None, axis=1)
            #plot lines
            for i in range(0, len(gdf)):
                ls = LineString([gdf["geometry"][i], self.movShapWM.loc[self.movShapWM[
                    "desc_zona"] == target].geometry.centroid.item()])
                ax.plot(*ls.xy, color = "green")

            #plot boundaries (do it again without points)
            vmin = currGroupInside[filter_col].sum(axis = 1).min()
            vmax = currGroupInside[filter_col].sum(axis = 1).max()
            step = (vmax-vmin)/10
            if vmax!=vmin:
                ticks = np.arange(vmin,vmax+step,step).astype(int)
            else:
                ticks = np.arange(10)
            currGroupInside = pd.merge(self.movShapWM,currGroupInside, how="inner", 
                                    left_on="desc_zona", right_on="ZONA_ORIG", suffixes=('', '_y'))
            currGroupInside.plot(ax = ax, alpha=0.5, edgecolor='k',linewidth = 5, 
                                column = currGroupInside[filter_col].sum(axis = 1), cmap = 'Reds', legend = True,
                                legend_kwds={'orientation': "horizontal","location": "top","shrink":0.87,
                                    "ticks": ticks} )
            
            # currGroupInside.apply(lambda x: 
            #                     ax.annotate(#Put name of centres
            #                         text=x['desc_zona'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]+500), ha='center',fontsize=15) 
            #                     if (x.desc_zona!= "MILANO 2")else None, axis=1)
            # currGroupInside.apply(lambda x: 
            #                     ax.annotate(#Put name of centres (Problem Milano2)
            #                         text=x['desc_zona'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]+1200), ha='center',fontsize=15) 
            #                     if (x.desc_zona== "MILANO 2")else None, axis=1)
            inPic = "static/Maps/"+uid+"/inside"+movType+".png"
            ax.figure.savefig(inPic)
            img.append(inPic+"?rand=" + str(random.randint(0, 1000)))
        
        return img

    # def getMovPosition(self, position):
    #     try:
    #         lat,lon = position.split("$")
    #         lat = float(lat)
    #         lon = float(lon)
    #     except:
    #         return
    #     point = gpd.points_from_xy([lon], [lat])
    #     pos = self.movShap[self.movShap.contains(point[0])]["desc_zona"].values[0]
    #     return pos