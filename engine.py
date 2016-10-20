#GIST_START engine.py
import logging
import sys

from pyspark import SparkConf, SparkContext, SQLContext
from pyspark.mllib.recommendation import ALS
from pyspark.sql.types import StructType, StructField, StringType, FloatType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CLOUDSQL_INSTANCE_IP = sys.argv[1]
CLOUDSQL_DB_NAME = sys.argv[2]
CLOUDSQL_USER = sys.argv[3]
CLOUDSQL_PWD = sys.argv[4]

conf = SparkConf().setAppName('Movie Recommender') \
    .set('spark.driver.memory', '6G') \
    .set('spark.executor.memory', '4G') \
    .set('spark.python.worker.memory', '4G')
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

TABLE_RATINGS = 'RATINGS'
TABLE_MOVIES = 'MOVIES'
TABLE_RECOMMENDATIONS = 'RECOMMENDATIONS'

jdbcDriver = 'com.mysql.jdbc.Driver'
jdbcUrl = 'jdbc:mysql://%s:3306/%s?user=%s&password=%s' % (
    CLOUDSQL_INSTANCE_IP, CLOUDSQL_DB_NAME, CLOUDSQL_USER, CLOUDSQL_PWD)

logger.info("Loading Datasets from MySQL...")

dfRates = sqlContext.read.format('jdbc') \
    .option('useSSL', False) \
    .option("url", jdbcUrl) \
    .option("dbtable", TABLE_RATINGS) \
    .option("driver", 'com.mysql.jdbc.Driver') \
    .load()

dfMovies = sqlContext.read.format('jdbc') \
    .option('useSSL', False) \
    .option("url", jdbcUrl) \
    .option("dbtable", TABLE_MOVIES) \
    .option("driver", 'com.mysql.jdbc.Driver') \
    .load()

dfRates.registerTempTable('Rates')

sqlContext.cacheTable('Rates')

logger.info("Datasets Loaded...")

rank = 8
seed = 5L
iterations = 10
regularization_parameter = 0.1

logger.info("Training the ALS model...")
model = ALS.train(dfRates.rdd.map(lambda r: (int(r[0]), int(r[1]), r[2])).cache(), rank=rank, seed=seed,
                  iterations=iterations, lambda_=regularization_parameter)
logger.info("ALS model built!")

# Calculate all predictions
predictions = model.recommendProductsForUsers(10) \
    .flatMap(lambda pair: pair[1]) \
    .map(lambda rating: (rating.user, rating.product, rating.rating))

schema = StructType([StructField("userId", StringType(), True), StructField("movieId", StringType(), True),
                     StructField("prediction", FloatType(), True)])

dfToSave = sqlContext.createDataFrame(predictions, schema)
dfToSave.write.jdbc(url=jdbcUrl, table=TABLE_RECOMMENDATIONS, mode='overwrite')
#GIST_END
