CREATE TABLE RATINGS (
userId varchar(255),
movieId varchar(255),
rating float,
PRIMARY KEY (userId, movieId)
FOREIGN KEY (movieId) REFERENCES MOVIES(id)
);