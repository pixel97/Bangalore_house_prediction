
function getBathValue(){
	var uiBathrooms = document.getElementsByName("uiBathrooms");
	for(var i in uiBathrooms){
		if(uiBathrooms[i].checked){
			return parseInt(i)+1;
		}
	}
	return -1
}

function getbhkValue(){
	var uiBHK = document.getElementsByName("uiBHK");
	for(var i in uiBHK){
		if(uiBHK[i].checked){
			return parseInt(i)+1;
		}
	}
	return -1
	
}



function onClickedEstimatePrice(){
	
	console.log("Estimate price button clicked");
	var sqft = document.getElementById("uiSqft");
	var bhk = getbhkValue();
	var bathrooms = getBathValue();
	var location = document.getElementById("uiLocations");
	var bestPrice = document.getElementById("uiEstimatedPrice");
	var url = "/api/predict_home_price"; 
  $.post(url, {
      total_sqft: parseFloat(sqft.value),
      bhk: bhk,
      bath: bathrooms,
      location: location.value
  },function(data, status) {
      console.log(data.estimated_price);
      bestPrice.innerHTML = "<h2>" + data.estimated_price.toString() + " Lakh</h2>";
      console.log(status);
  });
	
	
	
}



function onPageLoad(){
	
	console.log("Loading successful");
	var url = "/api/get_location_names";
	$.get(url,function(data,status){
		console.log("Got response from get_location_names");
		if(data){
			var locations = data.locations;
			var uiLocations = document.getElementById("uiLocations");
			$('#uiLocations').empty();
			for(var i in locations){
				var opt = new Option(locations[i]);
				$('#uiLocations').append(opt);
			}
		}
	
});

}

window.onload = onPageLoad;