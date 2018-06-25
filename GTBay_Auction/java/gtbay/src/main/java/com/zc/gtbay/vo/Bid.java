package com.zc.gtbay.vo;

import java.math.BigDecimal;
import java.sql.Timestamp;

public class Bid {
    private int bidId;
    private int itemId;
    private BigDecimal amount;
    private int userId;
    private Timestamp bidTime;
    private String username;

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public int getBidId() {
        return bidId;
    }

    public void setBidId(int bidId) {
        this.bidId = bidId;
    }

    public int getItemId() {
        return itemId;
    }

    public void setItemId(int itemId) {
        this.itemId = itemId;
    }

    public BigDecimal getAmount() {
        return amount;
    }

    public void setAmount(BigDecimal amount) {
        this.amount = amount;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public Timestamp getBidTime() {
        return bidTime;
    }

    public void setBidTime(Timestamp bidTime) {
        this.bidTime = bidTime;
    }
}
