CREATE TABLE RECOMMENDATIONS(
userId varchar(255),
movieId varchar(255),
prediction float,
PRIMARY KEY (userId, movieId),
FOREIGN KEY (userId,movieId) REFERENCES RATINGS(userId,movieId)
);