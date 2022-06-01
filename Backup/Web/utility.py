import math
import random
import pandas as pd
import geopandas as gpd
import plotly.express as px
from os import path
import matplotlib.pyplot as plt
import contextily as cx
from shapely.geometry import LineString
import numpy as np
import osmnx as ox

class utility:
    def __init__(self,staticFold):
        self.loadDatasetItaly(staticFold)
        self.loadDatasetMilan(staticFold)

    def loadDatasetMilan(self,staticFold):
        #Load shapefile Municipi
        self.municipi = gpd.read_file(path.join(staticFold, 'Dataset/Municipi/Municipi.shp')).sort_values(by=['MUNICIPIO']).reset_index().drop(["index"],axis = 1)
        #Load shapefile NIL
        nildf =  pd.read_csv(path.join(staticFold, 'Dataset/nilComplete.csv'))
        self.nil = gpd.GeoDataFrame(
            nildf.loc[:, [c for c in nildf.columns if c != "geometry"]],
            geometry=gpd.GeoSeries.from_wkt(nildf["geometry"]))
        #Load dataset population
        self.popMilano = pd.read_csv(path.join(staticFold, 'Dataset/popMilanComplete.csv'))
        #Load prices
        self.prices =  gpd.read_file(path.join(staticFold, 'Dataset/pricesTot.csv'),GEOM_POSSIBLE_NAMES="geometry",KEEP_GEOM_COLUMNS="NO")
        #Load shapefile CAP
        self.capZone = gpd.read_file(path.join(staticFold, 'Dataset/banchedati-cap-zone-demo-database')).sort_values(by=['CAP']).reset_index().drop(["index"],axis = 1)
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


    def plotItalyCluster(self,ageGroup,gender,maritalStatus,income):#TODO
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
            curr = curr.loc[curr["AvgIncome"] == takeClosest(curr,income)]#Text of current cluster
            selectedClusterPoints = self.df.loc[self.df["cluster"] == curr["cluster"].values[0]]
            print(curr)
            ...#Draw map,create and return
        else:
            return #No clusters found
        #Careful with -1 if use HDBSCAN

    def plotConnections(self):
        return "static/Maps/transportMilan.html"

    def plotMunicipi(self,ageGroup,gender,maritalStatus,numComp):
        considered = self.popMilano.loc[(self.popMilano["Classe_eta_capofamiglia"] == ageGroup)&
                        (self.popMilano["Genere_capofamiglia"] == gender)&
                        (self.popMilano["Tipologia_familiare"] == maritalStatus)&
                        (self.popMilano["Numero_componenti"] == numComp)]
        considered = considered.sort_values(by=['Municipio']).reset_index().drop(["index"],axis = 1)# Sort 
        if len(considered) != 9 or considered.empty: #Need to have data for all Municipi
            return
        fig = px.choropleth_mapbox(
            self.municipi,
            geojson=self.municipi,
            locations=self.municipi.index,
            hover_data =[self.municipi.MUNICIPIO], #Other written
            opacity =0.5,
            color = considered["Frq"], #Color zones
            center=dict(lat=45.45, lon=9.18),
            mapbox_style="open-street-map",
            zoom=11,
            height = 600,
            width = 600,
            labels={'MUNICIPIO':'Municipio','Frq':'Numero', 'color': 'Quantità'},
            color_continuous_scale = "peach",
            title = 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp)
        )
        fig.update_geos(fitbounds="locations", visible=True)
        url = "static/Maps/municipiGroup.html"
        fig.write_html(url)
        return url

    def plotNIL(self,ageGroup,gender,maritalStatus,numComp):
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
            fig = px.choropleth_mapbox(
                self.nil,
                geojson=self.nil,#TODO ROMPE TUTTO
                locations=self.nil.index,
                hover_data =[self.nil.NIL], #Bold name
                opacity =0.5,
                color = consideredNIL["RelFrq"], #Color zones
                center=dict(lat=45.45, lon=9.18),
                mapbox_style="open-street-map",
                zoom=11,
                height = 600,
                width = 600,
                labels={'NIL':'Quartiere','RelFrq':'Quantità', 'color': 'Quantità'},
                color_continuous_scale = "peach",
                title = 'Numero di %s, %s, %s, con famiglie da %s componenti' %(maritalStatus,gender,ageGroup,numComp)
            )
            fig.update_geos(fitbounds="locations", visible=True)
            url = "static/Maps/NILGroup.html"
            fig.write_html(url)
            return url
        else:
            return 

    def plotHousePrices(self):
        return "static/Maps/pricesMilan.html"

    def plotIncomes(self,income):
        consideredRed = self.redditi.loc[(self.redditi["Redditi e variabili IRPEF"] == income)]
        if consideredRed.empty:
            return
        consideredRed = consideredRed.sort_values(by=['CAP']).reset_index().drop(["index"],axis = 1)# Sort 
        if 'Frequenza' in income:
            label = 'Frequenza'
        else:
            label = 'Ammontare'
        fig = px.choropleth_mapbox(
            self.capZone,
            geojson=self.capZone,
            locations=self.capZone.index,
            hover_name =self.capZone.CAP, #Bold name
            opacity =0.5,
            color = consideredRed.Valori, #Color zones
            center=dict(lat=45.45, lon=9.18),
            mapbox_style="open-street-map",
            zoom=11,
            height = 600,
            width = 600,
            labels={'color':label,'AREA':'area'},
            title = income
        )
        fig.update_geos(fitbounds="locations", visible=True)

        url = "static/Maps/IncomeSelected.html"
        fig.write_html(url)
        return url
    
    def plotStores(self,position,storeType,storeName,storeBrand):
        selectedZone = self.movShap.loc[self.movShap["desc_zona"] == "MILANO 1"]#TODO position
        typeOfShop = ox.geometries.geometries_from_polygon(selectedZone["geometry"].values[0], tags = { 'shop':storeType})
        typeOfShop = typeOfShop.to_crs(epsg=3857)
        ax = typeOfShop.plot(figsize=(10, 10), alpha=0.5, edgecolor='k', color = "b")
        cx.add_basemap(ax)
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
        img = "static/Maps/stores.png"
        ax.figure.savefig(img)
        return img+"?rand=" + str(random.randint(0, 1000))


    def plotMovements(self,position):
        target = "MILANO 1"#TODO
        provinceOfTarget = "MI"
        movTarget = self.mov.loc[self.mov["ZONA_DEST"]== target]#Keep Only destination target
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
            images.append(self.plotMov(currGroupOutside,currGroupInside,target,movType,filter_col))
        
        return [item for sublist in images for item in sublist]#flat list of images

    def plotMov(self,currGroupOutside,currGroupInside,target, movType,filter_col):
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
            cx.add_basemap(ax)
            plt.axis('off')
            ax.set_title("Spostamenti verso " + target + ", tipo: " + movType + " da fuori provincia")

            #Print non-region data (plot is too big)
            for s in out:
                img.append(s)#Append where are movements from outside (not img just string)

            self.movShapWM.loc[self.movShapWM["desc_zona"] == target].apply(lambda x: ax.annotate(
                text=x['desc_zona'], xy=x.geometry.centroid.coords[0], ha='center',fontsize=15), axis=1);
            gdf.apply(lambda x: ax.annotate(#Annotate how many moves
                text=int(x[filter_col].sum()), xy=x.geometry.centroid.coords[0], ha='center',fontsize=20), axis=1);
            #plot lines
            for i in range(0, len(gdf)-1):
                ls = LineString([gdf["geometry"][i], self.movShapWM.loc[self.movShapWM[
                    "desc_zona"] == target].geometry.centroid.item()])
                ax.plot(*ls.xy, color = "green")
            outPic = "static/Maps/outside" + movType +".png"   
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
            cx.add_basemap(ax)
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
            ticks = np.arange(vmin,vmax+step,step).astype(int)
            currGroupInside = pd.merge(self.movShapWM,currGroupInside, how="inner", 
                                    left_on="desc_zona", right_on="ZONA_ORIG", suffixes=('', '_y'))
            currGroupInside.plot(ax = ax, alpha=0.5, edgecolor='k',linewidth = 5, 
                                column = currGroupInside[filter_col].sum(axis = 1), cmap = 'Reds', legend = True,
                                legend_kwds={'orientation': "horizontal","location": "top","shrink":0.87,
                                    "ticks": ticks} )
            
            currGroupInside.apply(lambda x: 
                                ax.annotate(#Put name of centres
                                    text=x['desc_zona'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]+500), ha='center',fontsize=15) 
                                if (x.desc_zona!= "MILANO 2")else None, axis=1)
            currGroupInside.apply(lambda x: 
                                ax.annotate(#Put name of centres (Problem Milano2)
                                    text=x['desc_zona'], xy=(x.geometry.centroid.coords[0][0],x.geometry.centroid.coords[0][1]+1200), ha='center',fontsize=15) 
                                if (x.desc_zona== "MILANO 2")else None, axis=1)
            inPic = "static/Maps/inside"+movType+".png"
            ax.figure.savefig(inPic)
            img.append(inPic+"?rand=" + str(random.randint(0, 1000)))
        
        return img