package com.zc.gtbay.common;

import com.alibaba.druid.pool.DruidDataSource;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.jdbc.core.JdbcTemplate;

import javax.sql.DataSource;


@Configuration
public class MysqlConfiguration {
    @Autowired
    private MysqlProperty mysqlProperty;

    @Bean
    public DataSource dataSource(){
        DruidDataSource dataSource = new DruidDataSource();
        dataSource.setDriverClassName(mysqlProperty.getClassName());
        dataSource.setUrl(mysqlProperty.getUrl());
        dataSource.setUsername(mysqlProperty.getUserName());
        dataSource.setPassword(mysqlProperty.getPassword());
        dataSource.setMaxActive(20);
        dataSource.setMinIdle(5);
        return dataSource;
    }

    @Bean
    public JdbcTemplate jdbcTemplate(){
        return new JdbcTemplate(dataSource());
    }

}
