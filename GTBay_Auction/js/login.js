/**
 * If the Login button is clicked
 */
$('#login').click((e) => {
    //console.log(username);
    //console.log(document.getElementById("password").value);
    e.preventDefault();

    var username = document.getElementById("username").value;
    var password = document.getElementById("password").value;
    //validate username and password are filled
    if (username == "") {
      alert("Username must be filled!");
      return false;
    }
    else if (password == "") {
        alert("Password must be filled!");
        return false;
    }
    // Call back-end, set sessionStorage for userID, username and userPosition;
    // open main menu according to regular user or admin user.
    else {
        $.post("http://localhost:12301/user/login",
            {username: username, password: password},
            function(result){
                console.log(result);

                if (result.code == 200) {
                    sessionStorage.setItem("currentUser", username);
                    sessionStorage.setItem("currentUserID", result.data.userId);

                    if (result.data.position == null) {
                        sessionStorage.setItem("currentUserPosition", "regular");
                        window.open('http://localhost:8888/Phase3/html/main_menu.html', "_self");
                    }
                    else {
                        sessionStorage.setItem("currentUserPosition", result.data.position);
                        window.open('http://localhost:8888/Phase3/html/main_menu_admin.html', "_self");
                    }

                    console.log(sessionStorage.getItem("currentUserID"));

                }
                else {
                    alert(result.msg);
                }

            });

    }
})




