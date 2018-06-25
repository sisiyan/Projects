package com.zc.gtbay.controller;

import com.zc.gtbay.service.UserService;
import com.zc.gtbay.vo.AdminUser;
import com.zc.gtbay.vo.CategoryReport;
import com.zc.gtbay.vo.Result;
import com.zc.gtbay.vo.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.HashMap;
import java.util.List;
import java.util.Map;


@CrossOrigin("http://localhost:8888")
@RestController
@RequestMapping(value = "/user")
public class UserController {
    @Autowired
    UserService userService;

    @RequestMapping(value = "/register", method = RequestMethod.POST)
    public Result register(@RequestParam(value = "first_name") String firstName,
                           @RequestParam(value = "last_name") String lastName,
                           @RequestParam(value = "username") String userName,
                           @RequestParam(value = "password") String password){

        Result result = new Result();
        User user = userService.findUserByUsername(userName);
        if(user == null){
            User user1 = new User();
            user1.setUsername(userName);
            user1.setPassword(password);
            user1.setFirstName(firstName);
            user1.setLastName(lastName);
            user1.setRole(0);
            int row = userService.insertUser(user1);
            if(row > 0){
                result.setCode(200);
            }else{
                result.setCode(400);
                result.setMsg("database internal error");
            }
        }else{
            result.setCode(400);
            result.setMsg("user already exists");
        }
        return result;

    }


    @RequestMapping(value = "/login", method = RequestMethod.POST)
    public Result login(@RequestParam(value = "username") String username,
                        @RequestParam(value = "password") String password){
        Result result = new Result();
        User user = userService.findUserByUsername(username);
        if(user == null){
            result.setCode(400);
            result.setMsg("user not exists");
        }
        else {
            if (user.getPassword().equals(password)) {
                Map<String, Object> data = new HashMap<>();
                data.put("userId", user.getUserID());
                result.setCode(200);
                if (user.getRole() == 1) {
                    AdminUser adminUser = userService.findAdminUserById(user.getUserID());
                    data.put("position", adminUser.getPosition());
                }
                result.setData(data);
            }
            else {
                result.setCode(400);
                result.setMsg("password not correct");
            }

        }
        return result;
    }

    @RequestMapping(value = "/user_report", method = RequestMethod.GET)
    public Result categoryReport(){
        Result result = new Result();
        List<User> users = userService.userReport();
        Map<String, Object> data = new HashMap<>();
        data.put("userReports", users);
        result.setCode(200);
        result.setData(data);
        return result;

    }
}
