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
        self.df= pd.read_csv(path.join(staticFold, "Dataset/clusteringItalia.csv"))
        clusterNames = []
        for i in range(0,max(self.df.cluster)+1):#For each cluster
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
        #Load "shapefile" italy for plotting
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

    def plotItalyCluster(self,ageGroup,gender,maritalStatus,income,uid):
        curr = self.clusterTypes.loc[(self.clusterTypes["AgeGroup"] == ageGroup) & 
                                        (self.clusterTypes["Gender"] == gender) & 
                                        (self.clusterTypes["MaritalStatus"] == maritalStatus)]
        def takeClosest(curr, targetIncome):
            closestIncome = curr.iloc[:1]["AvgIncome"].values[0]
            for i in range(1, curr.shape[0]):    
                currInc = curr.iloc[i:i+1]["AvgIncome"].values[0]
                if abs(currInc - targetIncome) < abs(closestIncome - targetIncome):
                    closestIncome = currInc
            return closestIncome
        if curr.shape[0]!=0:
            #Select points
            curr = curr.loc[curr["AvgIncome"] == takeClosest(curr,income)]#Text of current cluster
            selectedClusterPoints = self.df.loc[self.df["cluster"] == curr["cluster"].values[0]]

            clf = IsolationForest( random_state=0).fit_predict(selectedClusterPoints[["lng","lat"]].values)#Schema a blochi pptx

            lonC, latC = self.calculateWeightedMean(selectedClusterPoints["lat"], 
                        selectedClusterPoints["lng"], selectedClusterPoints["populPerKm2"],clf)
            circle = self.kmRadius((lonC,latC))
            closest = self.closest_point((lonC,latC), list(zip(selectedClusterPoints["lng"], selectedClusterPoints["lat"])))#Get closest point from picked

            xp = np.append(selectedClusterPoints["lng"].values,np.append(circle[:,0],lonC))
            yp = np.append(selectedClusterPoints["lat"].values,np.append(circle[:,1],latC))
            zp = np.append(selectedClusterPoints["populPerKm2"].values,np.append(np.zeros(len(circle[:,1])),0))
            color = np.append(clf,np.append(np.zeros(len(circle[:,1])),5))
            scatt = []
            #Draw Italy from csv
            for i in range (0,len(self.italy),2):
                x = self.italy.loc[i].values
                x = x[~np.isnan(x)]
                y = self.italy.loc[i+1].values
                y = y[~np.isnan(y)]
                scatt.append(go.Scatter3d(x=x, y=y, z=np.zeros(len(x)),showlegend=False,
                    mode='lines',
                    marker=dict(
                            color="lime",                # set color to an array/list of desired values
                            colorscale='Viridis',   # choose a colorscale
                            opacity=0.8,
                        )
                    )
                )
            #Add data cluster
            scatt.append(go.Scatter3d(x=xp, y=yp, z=zp,showlegend=False,
                                        mode='markers',hovertext =selectedClusterPoints["Territorio"],
                                        marker=dict(
                                                color=color,                # set color to an array/list of desired values
                                                colorscale='Viridis',   # choose a colorscale
                                                opacity=0.8,
                                            )
                                        )
                        )

            text = str(curr.values[0][0]) + " " + str(curr.values[0][1]) + " " + str(curr.values[0][2]) + " Redditi: " + str(curr.values[0][3]) + " Numero medio di componenti: " + str(curr.values[0][4])

            text += '\n' + 'Center is located in: %s' % (selectedClusterPoints.loc[(selectedClusterPoints["lng"] == closest[0])&
                                        (selectedClusterPoints["lat"] == closest[1])]["Territorio"].values[0])
            camera = dict(
                eye=dict(x=0, y=0, z=1),
                up=dict(x=0, y=1, z=0)
            )
            fig = go.Figure(data=scatt)
            fig.update_layout(scene_camera=camera)

            url = "static/Maps/"+uid+"/clusterItaly.html"
            fig.write_html(url)
            return [text,url]
        else:
            return #No clusters found
        #Careful with -1 if use HDBSCAN

    def plotConnections(self):
        return "static/Maps/transportMilan.html"

    def plotMunicipi(self,ageGroup,gender,maritalStatus,numComp,uid):
        considered = self.popMilano.loc[(self.popMilano["Classe_eta_capofamiglia"] == ageGroup)&
                        (self.popMilano["Genere_capofamiglia"] == gender)&
                        (self.popMilano["Tipologia_familiare"] == maritalStatus)&
                        (self.popMilano["Numero_componenti"] == numComp)]
        considered = considered.sort_values(by=['Municipio']).reset_index().drop(["index"],axis = 1)# Sort 
        if len(considered) != 9 or considered.empty: #Need to have data for all Municipi
            return
        fig, ax = plt.subplots(figsize=(15, 15))
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
        # fig = px.choropleth_mapbox(
        #     self.municipi, municipi not in crs different
        #     geojson=self.municipi,
        #     locations=self.municipi.index,
        #     hover_data =[self.municipi.MUNICIPIO], #Other written
        #     opacity =0.5,
        #     color = considered["Frq"], #Color zones
        #     center=dict(lat=45.45, lon=9.18),
        #     mapbox_style="open-street-map",
        #     zoom=11,
        #     height = 600,
        #     width = 600,
        #     labels={'MUNICIPIO':'Municipio','Frq':'Numero', 'color': 'Quantità'},
        #     color_continuous_scale = "peach",
        #     title = 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp)
        # )
        # fig.update_geos(fitbounds="locations", visible=True)
        # url = "static/Maps/"+uid+"/municipiGroup.html"
        # fig.write_html(url)
        return img+"?rand=" + str(random.randint(0, 1000))

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
            return img+"?rand=" + str(random.randint(0, 1000))
            # fig = px.choropleth_mapbox( no crs
            #     self.nil,
            #     geojson=self.nil,
            #     locations=self.nil.index,
            #     hover_data =[self.nil.NIL], #Bold name
            #     opacity =0.5,
            #     color = consideredNIL["RelFrq"], #Color zones
            #     center=dict(lat=45.45, lon=9.18),
            #     mapbox_style="open-street-map",
            #     zoom=11,
            #     height = 600,
            #     width = 600,
            #     labels={'NIL':'Quartiere','RelFrq':'Quantità', 'color': 'Quantità'},
            #     color_continuous_scale = "peach",
            #     title = 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp)
            # )
            # fig.update_geos(fitbounds="locations", visible=True)
            # url = "static/Maps/"+uid+"/NILGroup.html"
            # fig.write_html(url)
            # return url
        else:
            return 

    def plotHousePrices(self):
        return "static/Maps/pricesMilan.html"

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
        return img+"?rand=" + str(random.randint(0, 1000))
        # fig = px.choropleth_mapbox(
        #     self.capZone,
        #     geojson=self.capZone,
        #     locations=self.capZone.index,
        #     hover_name =self.capZone.CAP, #Bold name
        #     opacity =0.5,
        #     color = consideredRed.Valori, #Color zones
        #     center=dict(lat=45.45, lon=9.18),
        #     mapbox_style="open-street-map",
        #     zoom=11,
        #     height = 600,
        #     width = 600,
        #     labels={'color':label,'AREA':'area'},
        #     title = income
        # )
        # fig.update_geos(fitbounds="locations", visible=True)

        # url = "static/Maps/"+uid+"/incomeSelected.html"
        # fig.write_html(url)
        # return url
    
    def plotStores(self,position,storeType,storeName,storeBrand,uid):
        selectedZone = self.movShap.loc[self.movShap["desc_zona"] == position]
        typeOfShop = ox.geometries.geometries_from_polygon(selectedZone["geometry"].values[0], tags = { 'shop':storeType})
        typeOfShop = typeOfShop.to_crs(epsg=3857)
        ax = typeOfShop.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', color = "b")
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
                    text=x['PROV_ORIG'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]+800), ha='center',fontsize=15),axis=1)
            outPic = "static/Maps/"+uid+"/outside" + movType +".png"   
            ax.figure.savefig(outPic)
            img.append(outPic+"?rand=" + str(random.randint(0, 1000)))    
                
        #Inside movements
        if currGroupInside.empty:
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

    def getMovPosition(self, position):
        try:
            lat,lon = position.split("$")
            lat = float(lat)
            lon = float(lon)
        except:
            return
        point = gpd.points_from_xy([lon], [lat])
        pos = self.movShap[self.movShap.contains(point[0])]["desc_zona"].values[0]
        return pos