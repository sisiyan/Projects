DROP DATABASE IF EXISTS `cs6400_sp18_team090`;
SET default_storage_engine=InnoDB;

CREATE DATABASE IF NOT EXISTS cs6400_sp18_team090
    DEFAULT CHARACTER SET utf8mb4
    DEFAULT COLLATE utf8mb4_unicode_ci;
USE cs6400_sp18_team090;


SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
SET AUTOCOMMIT = 0;
START TRANSACTION;
SET time_zone = "+00:00";

-- Tables:

CREATE TABLE AdminUser (
  adminID int(11) NOT NULL AUTO_INCREMENT,
  username varchar(32) NOT NULL,
  position varchar(32) NOT NULL,
  PRIMARY KEY (adminID),
  UNIQUE KEY username (username)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE `User` (
  userID int(8) NOT NULL AUTO_INCREMENT,
  username varchar(32) NOT NULL,
  first_name varchar(32) NOT NULL,
  last_name varchar(32) NOT NULL,
  password varchar(32) NOT NULL,
  role int(4) DEFAULT NULL,
  PRIMARY KEY (userID),
  UNIQUE KEY username (username)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE Item (
  itemID int(11) NOT NULL AUTO_INCREMENT,
  item_name varchar(32) NOT NULL,
  description varchar(512) NOT NULL,
  categoryID int(11) NOT NULL,
  conditionID int(11) NOT NULL,
  start_bid_price decimal(10,2) NOT NULL,
  minimum_sale_price decimal(10,2) NOT NULL,
  auction_start_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  auction_length int(11) NOT NULL,
  get_now_price decimal(10,2) DEFAULT NULL,
  return_accepted tinyint(1) NOT NULL,
  userID int(11) NOT NULL,
  auction_end int(11) DEFAULT '0',
  end_time timestamp NULL DEFAULT NULL,

  PRIMARY KEY (itemID),
  KEY userID (userID),
  KEY categoryID (categoryID)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;

CREATE TABLE Bid (
  bidID int(11) NOT NULL AUTO_INCREMENT,
  itemID int(11) NOT NULL,
  amount decimal(10,2) NOT NULL,
  userID int(11) NOT NULL,
  bid_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (bidID),
  KEY userID_bid_time (userID,bid_time),
  KEY itemID (itemID)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE Category (
  categoryID int(11) NOT NULL AUTO_INCREMENT,
  category_name varchar(32) NOT NULL,
  PRIMARY KEY (categoryID),
  UNIQUE KEY type_name (category_name)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE Conditions (
  conditionID int(11) NOT NULL AUTO_INCREMENT,
  condition_name varchar(32) NOT NULL,
  PRIMARY KEY (conditionID),
  UNIQUE KEY condition_name (condition_name)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


CREATE TABLE Rating (
  ratingID int(11) NOT NULL AUTO_INCREMENT,
  itemID int(11) NOT NULL,
  rate_time timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  comments varchar(512) DEFAULT NULL,
  stars int(11) NOT NULL,
  userID int(11) NOT NULL,
  PRIMARY KEY (ratingID),
  KEY userId_rateTme (userID,rate_time),
  KEY itemID (itemID)
) ENGINE=InnoDB DEFAULT CHARSET=latin1;


-- Create view `high_bid`

CREATE VIEW high_bid AS
SELECT b.userid, tmp.winner_price, tmp.itemID from bid b RIGHT JOIN (
SELECT itemID, max(amount) as winner_price FROM Bid
GROUP BY itemID) tmp on tmp.itemid = b.itemid and b.amount = tmp.winner_price;

-- Create view `winner`
CREATE VIEW winner AS
  SELECT
    `b`.`userID`         AS `userid`,
    `tmp`.`winner_price` AS `winner_price`,
    `tmp`.`itemID`       AS `itemID`
  FROM ((SELECT
           `cs6400_sp18_team090`.`bid`.`itemID`      AS `itemID`,
           max(`cs6400_sp18_team090`.`bid`.`amount`) AS `winner_price`
         FROM `cs6400_sp18_team090`.`bid`
         GROUP BY `cs6400_sp18_team090`.`bid`.`itemID`) `tmp` LEFT JOIN `cs6400_sp18_team090`.`bid` `b`
      ON (((`tmp`.`itemID` = `b`.`itemID`) AND (`b`.`amount` = `tmp`.`winner_price`))));
