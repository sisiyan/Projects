package com.zc.gtbay.vo;

import java.sql.Timestamp;

public class Rating {
    private int ratingId;
    private int itemId;
    private Timestamp rateTime;
    private String comments;
    private int stars;
    private int userId;
    private String username;

    public String getUsername() {
        return username;
    }

    public void setUsername(String username) {
        this.username = username;
    }

    public int getUserId() {
        return userId;
    }

    public void setUserId(int userId) {
        this.userId = userId;
    }

    public int getRatingId() {
        return ratingId;
    }

    public void setRatingId(int ratingId) {
        this.ratingId = ratingId;
    }

    public int getItemId() {
        return itemId;
    }

    public void setItemId(int itemId) {
        this.itemId = itemId;
    }

    public Timestamp getRateTime() {
        return rateTime;
    }

    public void setRateTime(Timestamp rateTime) {
        this.rateTime = rateTime;
    }

    public String getComments() {
        return comments;
    }

    public void setComments(String comments) {
        this.comments = comments;
    }

    public int getStars() {
        return stars;
    }

    public void setStars(int stars) {
        this.stars = stars;
    }
}
