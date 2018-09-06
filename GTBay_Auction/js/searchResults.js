var table = '';
var itemList = [1001, "GPS", "$100", "Kevin", "$101", "4/4/2018"];
var rows = 1;
var cols = 6;

table = "<tr>"+
        "<th> ID </th>"+
        "<th> Item Name </th>"+
        "<th> Current Bid </th>"+
        "<th> High Bidder </th>"+
        "<th> Get It Now Price </th>"+
        "<th> Auction Ends </th>"+
        "</tr>"
for(var r= 0; r < rows; r++) {
  table += '<tr>';
  for(var c = 0; c <cols; c++){
    if (c ==1) {
      table += '<td>' + '<a href="../html/item_search.html">'+itemList[c]+'</a>'+ '</td>';
    }
    else {
      table += '<td>' +itemList[c]+ '</td>';
    }
  }
  table += '</tr>';
}

document.write('<table border =1>' + table + '</table>');
