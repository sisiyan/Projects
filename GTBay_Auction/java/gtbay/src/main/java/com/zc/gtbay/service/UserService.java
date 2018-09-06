package com.zc.gtbay.service;

import com.zc.gtbay.vo.AdminUser;
import com.zc.gtbay.vo.User;

import java.util.List;

public interface UserService {
    User findUserByUsername(String username);
    int insertUser(User user);
    AdminUser findAdminUserById(int id);
    List<User> userReport();
}
