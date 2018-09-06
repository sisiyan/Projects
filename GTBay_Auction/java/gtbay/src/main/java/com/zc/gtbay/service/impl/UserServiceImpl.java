package com.zc.gtbay.service.impl;

import com.zc.gtbay.service.UserService;
import com.zc.gtbay.vo.AdminUser;
import com.zc.gtbay.vo.User;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Service;
import org.springframework.web.bind.annotation.CrossOrigin;

import java.sql.ResultSet;
import java.sql.SQLException;
import java.util.List;

@Service(value = "userService")
public class UserServiceImpl implements UserService {
    @Autowired
    JdbcTemplate jdbcTemplate;

    @Override
    public User findUserByUsername(String username) {
        String sql = "SELECT userID, username, first_name, last_name, password, role from User where username = ?";
        List<User> users =  jdbcTemplate.query(sql, new Object[]{username}, new RowMapper<User>() {
            @Override
            public User mapRow(ResultSet resultSet, int i) throws SQLException {
                User user = new User();
                user.setUserID(resultSet.getInt("userID"));
                user.setUsername(resultSet.getString("username"));
                user.setFirstName(resultSet.getString("first_name"));
                user.setLastName(resultSet.getString("last_name"));
                user.setPassword(resultSet.getString("password"));
                user.setRole(resultSet.getInt("role"));
                return user;
            }
        });
        return !users.isEmpty() ? users.get(0) :  null;
    }

    @Override
    public int insertUser(User user) {
        String sql = " INSERT INTO User (username, first_name, last_name, password, role) " +
                "values (?,?,?,?,?)";
        int row = jdbcTemplate.update(sql, user.getUsername(), user.getFirstName(),user.getLastName(),user.getPassword(),user.getRole());
        return row;
    }

    @Override
    public AdminUser findAdminUserById(int id) {
        String sql = "select adminid, username, position from adminuser where adminid = ?";
        List<AdminUser> adminUsers = jdbcTemplate.query(sql, new Object[]{id}, new RowMapper<AdminUser>() {

            @Override
            public AdminUser mapRow(ResultSet resultSet, int i) throws SQLException {
                AdminUser user = new AdminUser();
                user.setAdminID(resultSet.getInt("adminid"));
                user.setUsername(resultSet.getString("username"));
                user.setPosition(resultSet.getString("position"));
                return user;
            }
        });
        return adminUsers.isEmpty() ? null : adminUsers.get(0);
    }

    @Override
    public List<User> userReport() {
        String sql = "select userid, username, " +
                "(select count(1) from item i where i.userid =u.userid  )listed, " +
                "(select count(1) from rating r where r.userid =u.userid) rated, " +
                "(select count(1) from item i ,winner w where now()>end_time and i.userid = u.userid and w.itemid = i.itemid) sold, " +
                "(select count(1) from item i ,winner w where now()>end_time and w.userid = u.userid and w.itemid = i.itemid) purchased " +
                "from user u";
        List<User> list = jdbcTemplate.query(sql, new Object[]{}, new RowMapper<User>() {
            @Override
            public User mapRow(ResultSet resultSet, int i) throws SQLException {
                User user = new User();
                user.setUserID(resultSet.getInt("userid"));
                user.setUsername(resultSet.getString("username"));
                user.setListed(resultSet.getInt("listed"));
                user.setRated(resultSet.getInt("rated"));
                user.setPurchased(resultSet.getInt("purchased"));
                user.setSold(resultSet.getInt("sold"));
                return user;
            }
        });
        return list;
    }


}
