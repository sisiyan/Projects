/**
 * Check the input fields
 * If all valid, submit the item to database
 * @returns {boolean}
 */

function listItem() {
    var itemName = document.forms["itemForm"]["iname"].value;
    var description = document.forms["itemForm"]["dsp"].value;
    var category = document.getElementById("category_list");
    var categoryID = category.options[category.selectedIndex].value;
    var condition = document.getElementById("condition_list");
    var conditionID = condition.options[condition.selectedIndex].value;
    var start_price = document.forms["itemForm"]["start_price"].value;
    var minimum_sale = document.forms["itemForm"]["min_price"].value;
    var get_now_price = document.forms["itemForm"]["getnow_price"].value;
    var auctionEndIn = document.getElementById("auctionLength");
    var auctionLength = auctionEndIn.options[auctionEndIn.selectedIndex].value;

    var returnAcp = document.getElementById("returnCheck").checked ? 1 : 0;
    var userID = sessionStorage.getItem("currentUserID");
    console.log("start price" + start_price);
    console.log("min price" + minimum_sale);
    //console.log(minimum_sale);


    if (sessionStorage.getItem("currentUser") == null) {
        alert("You have to login first!");
        return false;
    }
    if (itemName == "") {
        alert("Item Name must be filled out");
        return false;
    }

    if (description == "") {
        alert("Item description must be filled out");
        return false;
    }
    if (isNaN(start_price) || parseFloat(start_price) <= 0) {
        alert("start price must be a positive number!");
        return false;
    }
    if (isNaN(minimum_sale)) {
        alert("Minimum sale price must be a positive number!");
    }
    else if (parseFloat(minimum_sale) < parseFloat(start_price)) {
        console.log("start price " + start_price);
        console.log("min price " + minimum_sale);
        alert("Minimum sale price must be no smaller than start bidding price!");
        return false;
    }

    if (isNaN(get_now_price)) {
        alert("Get it now price must be a number!");
        return false;
    }
    else if (get_now_price != "" && parseFloat(get_now_price) < parseFloat(minimum_sale)) {
        alert("Get it now price must be higher than the minimum_sale price!");
        return false;
    }

    else {
        console.log(userID);
        $.post("http://localhost:12301/trade/create",
            {item_name: itemName,
                description: description,
                category_id: categoryID,
                condition_id: conditionID,
                start_bid_price: start_price,
                minimum_sale_price: minimum_sale,
                auction_length: auctionLength,
                get_now_price: get_now_price,
                return_accepted: returnAcp,
                user_id: userID},
            function(result){
                console.log(result.code);
                if (result.code == 200) {
                    console.log(result.code);
                    alert("List item success!");
                    var position = sessionStorage.getItem("currentUserPosition");
                    if (position == "regular") {
                        window.open('http://localhost:8888/Phase3/html/main_menu.html', "_self");
                    }
                    else {
                        window.open('http://localhost:8888/Phase3/html/main_menu_admin.html', "_self");
                    }
                }
                else {
                    alert(result.msg);
                }

            });
    }
}