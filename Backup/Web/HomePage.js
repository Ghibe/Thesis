(function() { // avoid variables ending up in the global scope

document.getElementById("showMap").addEventListener('click', createMapItaly,false);

document.getElementById("showMapMilan").addEventListener('click', createMapsMilan,false);

document.getElementById("showTransportMilan").addEventListener('click', showTransportMilan,false);

document.getElementById("showMovementsMilan").addEventListener('click', movementsMilan,false);

document.getElementById("showStores").addEventListener('click', showStores,false);

      // create map in div 'map'
      function createMapItaly(){//TODO TEST
      var completed = true;
      var params = "";
        //See if all info set for clusters
      if(document.getElementById("ageGroup").value==null ||document.getElementById("ageGroup").value==""){
			  document.getElementById("errorIT").innerHTML = "Please select an appropriate age group";
			  completed = false;
		  }
      else{
        params+= "&ageGroup="
        params+= document.getElementById("ageGroup").value
      }

      if(document.getElementById("Gender").value==null ||document.getElementById("Gender").value==""){
			  document.getElementById("errorIT").innerHTML = "Please select an appropriate gender";
			  completed = false;
		  }
      else{
        params+= "&Gender="
        params+= document.getElementById("Gender").value
      }

      if(document.getElementById("maritalStatus").value==null ||document.getElementById("maritalStatus").value==""){
			  document.getElementById("errorIT").innerHTML = "Please select an appropriate marital status";
			  completed = false;
		  }
      else{
        params+= "&maritalStatus="
        params+= document.getElementById("maritalStatus").value
      }
      
      if(document.getElementById("avgIncome").value==null ||document.getElementById("avgIncome").value==""){
			  document.getElementById("errorIT").innerHTML = "Please insert an average income";
			  completed = false;
		  }
      else{
        params+= "&avgIncome="
        params+= document.getElementById("avgIncome").value
      }
      if (completed){
        document.getElementById("ItalyMap").innerHTML="";//Clean
        var request = new XMLHttpRequest();
        request.onreadystatechange = function() {
          if (request.readyState == XMLHttpRequest.DONE) {
            switch (request.status) {
            case 200:
              var url = JSON.parse(request.responseText);
              document.getElementById("ItalyMap").innerHTML='<object type="text/html" data=' + url + ' ></object>';
              break;
            case 204:
              document.getElementById("ItalyMap").innerHTML="No values found for selected cluster, please try with other specifics";
              break;
            default:
              document.getElementById("errorIT").innerHTML = "Unknown error";
            }
          }
        };//end callbackFunc
        request.open("POST", "./ItalyMap", true);
        request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        request.send(params);
        var lding= document.createElement('img');
        lding.src = "static/Loading_icon.gif";
        document.getElementById("ItalyMap").appendChild(lding)
      }
    }

    function showTransportMilan(){
      document.getElementById("errorMil").innerHTML = "";
      var request = new XMLHttpRequest();
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          switch (request.status) {
          case 200:
            var url = request.responseText;
            document.getElementById("transportMilan").innerHTML='<object type="text/html" data=' + url + ' ></object>';
            break;
          default:
            document.getElementById("errorMil").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("GET", "./ConnectionMilan", true);
			request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
			request.send();
    }


    function createMapsMilan(){
      var completed = true;
      var params = "";
        //See if all info set for clusters
      if(document.getElementById("ageGroupCity").value==null ||document.getElementById("ageGroupCity").value==""){
			  document.getElementById("errorMil").innerHTML = "Please select an appropriate age group";
			  completed = false;
		  }
      else{
        params+= "&ageGroup="
        params+= document.getElementById("ageGroupCity").value
      }

      if(document.getElementById("GenderCity").value==null ||document.getElementById("GenderCity").value==""){
			  document.getElementById("errorMil").innerHTML = "Please select an appropriate gender";
			  completed = false;
		  }
      else{
        params+= "&Gender="
        params+= document.getElementById("GenderCity").value
      }

      if(document.getElementById("maritalStatusCity").value==null ||document.getElementById("maritalStatusCity").value==""){
			  document.getElementById("errorMil").innerHTML = "Please select an appropriate marital status";
			  completed = false;
		  }
      else{
        params+= "&maritalStatus="
        params+= document.getElementById("maritalStatusCity").value
      }
      
      if(document.getElementById("incomeCity").value==null ||document.getElementById("incomeCity").value==""){
			  document.getElementById("errorMil").innerHTML = "Please insert an average income";
			  completed = false;
		  }
      else{
        params+= "&avgIncome="
        params+= document.getElementById("incomeCity").value
      }

      if(document.getElementById("avgNumCompCity").value==null ||document.getElementById("avgNumCompCity").value==""){
			  document.getElementById("errorMil").innerHTML = "Please insert the average number of components";
			  completed = false;
		  }
      else{
        params+= "&avgNumComp="
        params+= document.getElementById("avgNumCompCity").value
      }
      if (completed){
        document.getElementById("milanMuni").innerHTML="";//Clean
        document.getElementById("milanNIL").innerHTML="";
        document.getElementById("housePrices").innerHTML="";
        document.getElementById("capIncome").innerHTML="";
        document.getElementById("errorMil").innerHTML ="";
        var request = new XMLHttpRequest();
        request.onreadystatechange = function() {
          if (request.readyState == XMLHttpRequest.DONE) {
            switch (request.status) {
            case 200:
              var url = JSON.parse(request.responseText);
              //Always at least one
              if (url.length < 2 ){ //No data for municipalities/NIL and income (only house prices)
                document.getElementById("housePrices").innerHTML='<object type="text/html" data=' + url[0] + ' ></object>';

                document.getElementById("milanMuni").innerHTML="No data for selected group";
                document.getElementById("milanNIL").innerHTML="No data for selected group";
                document.getElementById("capIncome").innerHTML="No data for selected group";
              }
              else if (url.length < 3 ){ //No data for municipalities/NIL (house prices + incomes)
                document.getElementById("housePrices").innerHTML='<object type="text/html" data=' + url[0] + ' ></object>';
                document.getElementById("capIncome").innerHTML='<object type="text/html" data=' + url[1] + ' ></object>';

                document.getElementById("milanMuni").innerHTML="No data for selected group";
                document.getElementById("milanNIL").innerHTML="No data for selected group";
              }
              else{ //Have all
                document.getElementById("milanMuni").innerHTML='<object type="text/html" data=' + url[0] + ' ></object>';
                document.getElementById("milanNIL").innerHTML='<object type="text/html" data=' + url[1] + ' ></object>';
                document.getElementById("housePrices").innerHTML='<object type="text/html" data=' + url[2] + ' ></object>';
                document.getElementById("capIncome").innerHTML='<object type="text/html" data=' + url[3] + ' ></object>';
              }
              break;
            default:
              document.getElementById("errorMil").innerHTML = "Unknown error";
            }
          }
        };//end callbackFunc
        request.open("POST", "./MilanMaps", true);
        request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        request.send(params);
        var lding= document.createElement('img');
        lding.src = "static/Loading_icon.gif";
        document.getElementById("milanMuni").appendChild(lding)
      }
    }

    function movementsMilan(){
      //TODO target
      var request = new XMLHttpRequest();
      var divMov = document.getElementById("movements");
      request.onreadystatechange = function() {
        if (request.readyState == XMLHttpRequest.DONE) {
          switch (request.status) {
          case 200:
            var listImgs = JSON.parse(request.responseText);
            divMov.innerHTML = '';//Clear all children
            //Add each image and text
            for (var i=0; i<listImgs.length; i++){
              if (listImgs[i].includes("png")){
                var imgToAdd = document.createElement('img');
                imgToAdd.src = listImgs[i];
                divMov.appendChild(imgToAdd);
              }
              else{ //Text explaining where
                var textToAdd = document.createTextNode(listImgs[i])
                divMov.appendChild(textToAdd);
              }
            }
            //document.getElementById("storesImg").src = request.responseText;
            break;
          default:
            document.getElementById("errorStores").innerHTML = "Unknown error";
          }
        }
      };//end callbackFunc
      request.open("POST", "./MovementsMilan", true);
      request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
      request.send();
      var lding= document.createElement('img');
      lding.src = "static/Loading_icon.gif";
      divMov.appendChild(lding)
    }

    function showStores(){
      var completed = true;
      var params = "";

      if(document.getElementById("storeTypes").value==null ||document.getElementById("storeTypes").value==""){
			  document.getElementById("errorStores").innerHTML = "Please select a store type";
			  completed = false;
		  }
      else{
        params+= "&storeType="
        params+= document.getElementById("storeTypes").value
      }

      if (completed){
        var divMov = document.getElementById("movements");
        document.getElementById("errorStores").innerHTML="";//Clean
        document.getElementById("storesImg").src = "";
        params+= "&storeNames="
        params+= document.getElementById("storeNames").value
        params+= "&brands="
        params+= document.getElementById("brands").value
        //TODO POSITION
        var request = new XMLHttpRequest();
        request.onreadystatechange = function() {
          if (request.readyState == XMLHttpRequest.DONE) {
            switch (request.status) {
            case 200:
              divMov.innerHTML = '';//Clear all children
              document.getElementById("storesImg").src = request.responseText;
              break;
            default:
              document.getElementById("errorStores").innerHTML = "Unknown error";
            }
          }
        };//end callbackFunc
        request.open("POST", "./StoresMilan", true);
        request.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
        request.send(params);
        var lding= document.createElement('img');
        lding.src = "static/Loading_icon.gif";
        divMov.appendChild(lding)
      }
    }
  })();