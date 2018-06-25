
$('#registerButton').click((e) => {
    e.preventDefault();

    var firstName = document.forms[0].elements[0].value;
    var lastName = document.forms[0].elements[1].value;
    var username = document.forms[0].elements[2].value;
    var password = document.forms[0].elements[3].value;
    var confirm_password = document.forms[0].elements[4].value;

    if (firstName === "") {
        alert("First Name must be filled out");
        return false;
    }
    else if (lastName === "") {
        alert("Last Name must be filled out");
        return false;
    }
    else if (username === "") {
        alert("Username must be filled out");
        return false;
    }
    else if (password === "") {
        alert("Password must be filled out");
        return false;
    }
    else if (confirm_password === "") {
        alert("Confirm_password must be filled out");
        return false;
    }
    else if (password !== confirm_password) {
        alert("Password and Confirm password not match!");
        return false;
    }
    else {
        $.post("http://localhost:12301/user/register",
            {first_name: firstName, last_name: lastName, username: username, password: password},
            function(result) {
                console.log(result);
                if (result.code == 200) {
                    window.open('http://localhost:8888/Phase3/html/login.html', "_self");
                    alert("Registration success! Welcome to login!");
                }
                else {
                    alert(result.msg);
                    return false;
                }
            });
    }
})
