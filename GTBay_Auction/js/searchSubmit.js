/**
function findDetail(itemID) {
    console.log(itemID);

    var submit_url = "http://localhost:12301/trade/detail";

    // do the ajax
    $.ajax({
        type: "GET",
        url: submit_url,
        data: {item_id: itemId},
        success: function(res) {
            // if the captcha is good, submit the form
            console.log(res)
        }
    });
}
*/

var findDetail = function (itemID) {
    console.log(itemID);

    var submit_url = "http://localhost:12301/trade/detail";

    // do the ajax
    $.ajax({
        type: "GET",
        url: submit_url,
        data: {item_id: itemId},
        success: function(res) {
            // if the captcha is good, submit the form
            console.log(res)
        }
    });
}
function createTable(items) {
    var itemId_list = [];
    var name_list = [];
    var bid_list = [];
    var bidder_list = [];
    var getNow_list = [];
    var endTime_list = [];
    for (var i = 0; i < items.length; i++) {
        itemId_list.push(items[i].itemId);
        name_list.push(items[i].itemName);
        bid_list.push(items[i].winnerPrice);
        bidder_list.push(items[i].userName);
        getNow_list.push(items[i].getNowPrice);
        endTime_list.push(items[i].endTime);
    }


    var rows = items.length;
    var cols = 6;

    var table = "<tr>"+
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
            if (c == 0) {
                table += '<td>' +itemId_list[r]+ '</td>';
            }
            else if (c ==1) {
                table += '<td>' + '<a href = "../html/itemDescription.html" onclick="findDetail(itemId_list[r]);">' + name_list[r]+'</a>'+ '</td>';
            }
            else if (c == 2) {
                table += '<td>' +bid_list[r]+ '</td>';
            }
            else if (c == 3) {
                table += '<td>' +bidder_list[r]+ '</td>';
            }
            else if (c == 4) {
                table += '<td>' +getNow_list[r]+ '</td>';
            }
            else if (c == 5) {
                table += '<td>' +endTime_list[r]+ '</td>';
            }
        }
        table += '</tr>';
    }



    //window.open('../html/SearchResults.html', "searchResults");
    document.write('<table border =1>' + table + '</table>');
    //window.open('../html/SearchResults.html', "_blank");

    //window.open('../html/SearchResults.html', "_blank").document.write('<table border =1>' + table + '</table>');

    window.open('http://localhost:8888/Phase3/html/SearchResults.html', "_blank");
    write('<table border =1>' + table + '</table>');

}

function submitForm() {
    var keyword = document.forms["searchForm"]["keyword"].value;

    var category = document.getElementById("category_list");
    var categoryID = category.options[category.selectedIndex].value;
    var condition = document.getElementById("condition_list");
    var conditionID = condition.options[condition.selectedIndex].value;

    var minimum_sale = document.forms["searchForm"]["min_price"].value;
    var maximum_sale = document.forms["searchForm"]["max_price"].value;


    if (keyword == "") {
        alert("keyword must be filled out");
        return false;
    }

    else if (isNaN(minimum_sale)|| minimum_sale < 0) {
        alert("Minimum sale price must be a positive number!");
        return false;
    }

    else if (isNaN(maximum_sale)|| maximum_sale < minimum_sale) {
        alert("Maximum sale price must be equal to or higher than minimum sale price!");
        return false;
    }

    else {

        //document.getElementById("selected_cat").value= returnAcp;
        //window.open('http://localhost:8888/Phase3/html/SearchResults.html', "_blank");

        $(document).ready(function() {
            $.get("http://localhost:12301/trade/search",
                {keyword: keyword, category_id: categoryID, min_price: minimum_sale,
                    max_price: maximum_sale, condition_id: conditionID},
                function(result){
                    function findDetail(itemID) {
                        console.log(itemID);

                        var submit_url = "http://localhost:12301/trade/detail";

                        // do the ajax
                        $.ajax({
                            type: "GET",
                            url: submit_url,
                            data: {item_id: itemId},
                            success: function(res) {
                                // if the captcha is good, submit the form
                                console.log(res)
                            }
                        });
                    }


                    var items = result.data.item_list;

                    console.log(items);
                    createTable(items)

                });
        });


        /***
        $('#searchResults').click((e) => {
            console.log("getting results")
            e.preventDefault();

            $(document).ready(function() {
                $.get("http://localhost:12301/trade/search",
                    {keyword: keyword, category_id: categoryID, min_price: minimum_sale,
                        max_price: maximum_sale, condition_id: conditionID},
                    function(result){
                        var items = result.data.item_list;
                        //var newWindow = window.open('../html/SearchResults.html', "searchResults");
                        //newWindow.focus();

                        createTable(items);


                        console.log(items);

                    });
            });
        })
         */

    }
}

